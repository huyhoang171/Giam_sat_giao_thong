import asyncio
import cv2
import base64
import time
import os
import re
from unidecode import unidecode
import socket
import struct
from typing import List, Dict, Set, Optional, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import subprocess
import shutil
from datetime import timedelta, datetime
import json
import numpy as np

from fastapi import FastAPI, WebSocket, Depends, Request, HTTPException, status, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt

# --- Các import từ module trong dự án ---
from . import crud, models, schemas, auth
from .database import engine, get_db, SessionLocal
from .cv_processor import ViolationDetector

# ====================================================================
# === CẤU HÌNH (CONFIGURATION) ===
# ====================================================================
STREAM_RECEIVE_PORT = 9999
STREAM_RECEIVE_IP = "0.0.0.0"

# --- Khởi tạo ---
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Hệ thống Giám sát Giao thông")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

received_frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)


# ====================================================================
# === HÀM HELPER ===
# ====================================================================

def normalize_filename(name: str) -> str:
    safe_name = unidecode(name)
    safe_name = re.sub(r'\s+', '_', safe_name)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', safe_name)
    return safe_name.lower()


def save_video_to_db(location_id: int, video_path: str):
    temp_path = video_path.replace(".mp4", "_faststart.mp4")
    print(f"[*] Đang tối ưu hóa video: {video_path}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, "-c:v", "libx264",
                "-preset", "veryfast", "-movflags", "+faststart", temp_path
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        shutil.move(temp_path, video_path)
        print(f"[*] Tối ưu hóa thành công: {video_path}")
    except FileNotFoundError:
        print(
            "[!] LỖI: Lệnh 'ffmpeg' không được tìm thấy. Hãy đảm bảo ffmpeg đã được cài đặt và có trong PATH hệ thống.")
    except Exception as e:
        print(f"[!] Lỗi khi tối ưu hóa video bằng ffmpeg: {e}")

    db = SessionLocal()
    try:
        crud.create_video_recording(
            db,
            video=schemas.VideoRecordingCreate(video_path=video_path),
            location_id=location_id
        )
        print(f"[*] Đã lưu thông tin video '{video_path}' vào CSDL.")
    except Exception as e:
        print(f"[!] Lỗi khi lưu video vào CSDL: {e}")
    finally:
        db.close()


def save_frame_as_image(frame: np.ndarray, base_dir: str, filename_prefix: str) -> Optional[str]:
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{filename_prefix}.jpg")
    try:
        if frame is not None and frame.size > 0:
            cv2.imwrite(file_path, frame)
            return file_path
        return None
    except Exception as e:
        print(f"[!] Lỗi khi lưu ảnh {file_path}: {e}")
        return None


# ====================================================================
# === LỚP QUẢN LÝ STREAM ===
# ====================================================================
class StreamManager:
    def __init__(self):
        self.current_stream_task: Optional[asyncio.Task] = None
        self.current_location_id: Optional[int] = None
        self.current_location_name: str = "Không có"
        self.current_violation_roi: Optional[str] = None
        self.current_detection_roi: Optional[str] = None
        self.viewers: Set[WebSocket] = set()
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.detector = ViolationDetector()
        self.ai_detection_enabled: bool = False
        self.socket_server_task: Optional[asyncio.Task] = None

    async def _capture_loop(self):
        if self.current_location_id is None: return

        location_id = self.current_location_id
        print(f"[*] Bắt đầu capture cho vị trí {self.current_location_name} (ID: {location_id}) từ Raspberry Pi.")
        os.makedirs("static/videos", exist_ok=True)

        VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
        RECORDING_DURATION_SECONDS = 60
        location_name_for_file = normalize_filename(self.current_location_name)

        initial_frame = None
        while initial_frame is None:
            try:
                initial_frame = await asyncio.wait_for(received_frame_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                print("[!] Lỗi: Không nhận được frame từ Raspberry Pi. Đảm bảo Pi đang gửi dữ liệu.")
                await asyncio.sleep(1)
                continue

        VIDEO_HEIGHT, VIDEO_WIDTH, _ = initial_frame.shape
        FRAME_RATE = 20.0

        video_writer = None
        last_saved_time = time.time()
        video_filename = ""

        while self.current_location_id == location_id:
            try:
                frame = await asyncio.wait_for(received_frame_queue.get(), timeout=5.0)
                if frame is None:
                    print("[!] Không nhận được frame từ Raspberry Pi. Đang chờ frame mới...")
                    continue

                if video_writer is None or (time.time() - last_saved_time) >= RECORDING_DURATION_SECONDS:
                    if video_writer:
                        video_writer.release()
                        print(f"[*] Đã ghi xong file video: {video_filename}")
                        await asyncio.to_thread(save_video_to_db, location_id, video_filename)

                    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    video_filename = f"static/videos/{location_name_for_file}_{ts_str}.mp4"
                    video_writer = cv2.VideoWriter(video_filename, VIDEO_CODEC, FRAME_RATE, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    last_saved_time = time.time()

                if video_writer is not None:
                    video_writer.write(frame)

                processed_frame = frame.copy()

                if self.ai_detection_enabled:
                    violation_roi_coords = None
                    detection_roi_coords = None

                    if self.current_violation_roi and self.current_violation_roi != 'None':
                        try:
                            v_roi_parsed = tuple(map(int, self.current_violation_roi.split(',')))
                            if len(v_roi_parsed) == 4:
                                violation_roi_coords = v_roi_parsed
                            else:
                                violation_roi_coords = None
                        except (ValueError, IndexError):
                            violation_roi_coords = None

                    if self.current_detection_roi and self.current_detection_roi != 'None':
                        try:
                            d_roi_parsed = tuple(map(int, self.current_detection_roi.split(',')))
                            if len(d_roi_parsed) == 4:
                                detection_roi_coords = d_roi_parsed
                            else:
                                detection_roi_coords = None
                        except (ValueError, IndexError):
                            detection_roi_coords = None

                    if violation_roi_coords:
                        processed_frame, violations_data_list, new_vehicles_count = self.detector.process_frame(
                            frame, violation_roi_coords, detection_roi_coords
                        )
                        if new_vehicles_count > 0:
                            db = SessionLocal()
                            try:
                                crud.increment_detected_vehicles_count(db, location_id, new_vehicles_count)
                            finally:
                                db.close()

                    else:
                        violations_data_list = []
                        if detection_roi_coords:
                            x1, y1, x2, y2 = detection_roi_coords
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                            cv2.putText(processed_frame, "Detection ROI", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

                    if violations_data_list:
                        for violation_data in violations_data_list:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            unique_filename_prefix = f"{location_name_for_file}_{timestamp_str}"
                            overview_path = save_frame_as_image(violation_data["overview_frame"],
                                                                "static/violations/overview",
                                                                f"overview_{unique_filename_prefix}")
                            vehicle_path = save_frame_as_image(violation_data["vehicle_frame"],
                                                               "static/violations/vehicle",
                                                               f"vehicle_{unique_filename_prefix}")
                            license_plate_path = save_frame_as_image(violation_data["plate_frame"],
                                                                     "static/violations/plate",
                                                                     f"plate_{unique_filename_prefix}")
                            db_violation = schemas.ViolationCreate(
                                license_plate_info=violation_data["license_plate_info"],
                                vehicle_type=violation_data.get("vehicle_type"),
                                overview_image_path=overview_path,
                                vehicle_image_path=vehicle_path,
                                license_plate_image_path=license_plate_path
                            )
                            db = SessionLocal()
                            try:
                                crud.create_violation(db, db_violation, location_id)
                                print(f"[*] Đã lưu vi phạm {violation_data['license_plate_info']} ({violation_data.get('vehicle_type')}) vào CSDL.")
                            finally:
                                db.close()

                if self.viewers and not self.frame_queue.full():
                    await self.frame_queue.put(processed_frame)

            except asyncio.TimeoutError:
                print("[!] Timeout khi chờ frame từ Raspberry Pi trong _capture_loop. Đang chờ frame mới...")
                continue
            except Exception as e:
                print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH XỬ LÝ FRAME HOẶC GHI VIDEO: {e}")
                if 'frame' in locals() and self.viewers and not self.frame_queue.full():
                    await self.frame_queue.put(frame)
                await asyncio.sleep(0.1)

        if video_writer and video_writer.isOpened():
            video_writer.release()
            await asyncio.to_thread(save_video_to_db, location_id, video_filename)

        print(f"[*] Đã dừng capture cho vị trí {location_id}.")

    async def _broadcast_loop(self):
        while self.current_stream_task and not self.current_stream_task.done():
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                if frame is None: break
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret: continue

                jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                message_to_send = {
                    "type": "frame_update",
                    "data": jpg_as_text
                }
                await self.broadcast_message(json.dumps(message_to_send))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[!] Lỗi trong quá trình broadcast frame: {e}")
                await asyncio.sleep(0.1)

    async def broadcast_message(self, message: str):
        disconnected_clients = set()
        for ws in self.viewers:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected_clients.add(ws)
        if disconnected_clients: self.viewers -= disconnected_clients

    async def switch_stream(self, new_location_id: int):
        if self.current_location_id == new_location_id: return

        self.detector.reset()

        if self.current_stream_task:
            self.current_location_id = None
            self.current_stream_task.cancel()
            try:
                await self.current_stream_task
            except asyncio.CancelledError:
                pass

        while not self.frame_queue.empty(): self.frame_queue.get_nowait()
        while not received_frame_queue.empty(): received_frame_queue.get_nowait()

        db = SessionLocal()
        new_location = crud.get_location(db, new_location_id)
        db.close()

        if not new_location:
            print(f"[!] Lỗi: Vị trí {new_location_id} không tồn tại.")
            self.current_location_name = "Không có"
            self.current_location_id = None
            self.current_violation_roi = None
            self.current_detection_roi = None
            self.ai_detection_enabled = False
            return

        self.current_location_id = new_location.id
        self.current_location_name = new_location.name
        self.current_violation_roi = new_location.violation_roi
        self.current_detection_roi = new_location.detection_roi
        self.ai_detection_enabled = False

        capture_task = asyncio.create_task(self._capture_loop())
        broadcast_task = asyncio.create_task(self._broadcast_loop())
        self.current_stream_task = asyncio.gather(capture_task, broadcast_task)
        print(f"[*] Đã chuyển sang giám sát tại: {self.current_location_name} (ID: {self.current_location_id})")
        status_message = {"type": "status_update", "location_id": self.current_location_id,
                          "location_name": self.current_location_name, "violation_roi": self.current_violation_roi,
                          "detection_roi": self.current_detection_roi,
                          "ai_detection_enabled": self.ai_detection_enabled}
        await self.broadcast_message(json.dumps(status_message))

    async def add_viewer(self, websocket: WebSocket):
        await websocket.accept()
        self.viewers.add(websocket)
        print(f"[*] Một người xem đã kết nối. Tổng số người xem: {len(self.viewers)}")
        status_message = {"type": "status_update", "location_id": self.current_location_id,
                          "location_name": self.current_location_name, "violation_roi": self.current_violation_roi,
                          "detection_roi": self.current_detection_roi,
                          "ai_detection_enabled": self.ai_detection_enabled}
        await websocket.send_text(json.dumps(status_message))

    async def remove_viewer(self, websocket: WebSocket):
        self.viewers.discard(websocket)
        print(f"[*] Một người xem đã ngắt kết nối. Tổng số người xem: {len(self.viewers)}")


stream_manager = StreamManager()


# ====================================================================
# === SOCKET SERVER ĐỂ NHẬN DỮ LIỆU TỪ PI ===
# ====================================================================
async def handle_pi_connection(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[+] Client Raspberry Pi đã kết nối từ: {addr}")

    data_buffer = b""
    payload_size = struct.calcsize(">L")

    try:
        while True:
            while len(data_buffer) < payload_size:
                chunk = await reader.read(4096)
                if not chunk:
                    print(f"[*] Client {addr} đã ngắt kết nối.")
                    return
                data_buffer += chunk

            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data_buffer) < msg_size:
                chunk = await reader.read(msg_size - len(data_buffer))
                if not chunk:
                    print(f"[*] Client {addr} đã ngắt kết nối trong khi đọc dữ liệu.")
                    return
                data_buffer += chunk

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            np_data = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is not None and frame.size > 0:
                if not received_frame_queue.full():
                    await received_frame_queue.put(frame)
                else:
                    pass
            else:
                print("[!] Frame nhận được từ Pi không hợp lệ hoặc trống.")

    except asyncio.CancelledError:
        print(f"[*] Xử lý kết nối Pi cho {addr} bị hủy.")
    except Exception as e:
        print(f"[!!!] Lỗi khi xử lý kết nối Pi từ {addr}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[*] Kết nối với {addr} đã đóng.")


async def start_socket_server():
    print(f"[*] Khởi động Socket Server trên {STREAM_RECEIVE_IP}:{STREAM_RECEIVE_PORT} để nhận ảnh từ Pi...")
    server = await asyncio.start_server(
        handle_pi_connection, STREAM_RECEIVE_IP, STREAM_RECEIVE_PORT
    )
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f"[*] Đang phục vụ trên {addrs}")
    async with server:
        await server.serve_forever()


# ====================================================================
# === SỰ KIỆN STARTUP/SHUTDOWN CỦA SERVER ===
# ====================================================================
@app.on_event("startup")
async def startup_event():
    stream_manager.socket_server_task = asyncio.create_task(start_socket_server())

    db = SessionLocal()
    first_location = db.query(models.Location).first()
    db.close()
    if first_location:
        print("[*] Server khởi động. Tự động stream cho vị trí đầu tiên...")
        asyncio.create_task(stream_manager.switch_stream(first_location.id))
    else:
        print("[!] Cảnh báo: Không tìm thấy vị trí nào để stream trong CSDL.")


@app.on_event("shutdown")
async def shutdown_event():
    print("[*] Server tắt. Đang dừng stream hiện tại...")
    if stream_manager.current_stream_task:
        stream_manager.current_stream_task.cancel()
        await asyncio.gather(stream_manager.current_stream_task, return_exceptions=True)

    if stream_manager.socket_server_task:
        stream_manager.socket_server_task.cancel()
        try:
            await stream_manager.socket_server_task
        except asyncio.CancelledError:
            pass
    print("[*] Đã dừng stream và socket server.")


# ====================================================================
# === API ENDPOINTS VÀ WEBSOCKET ===
# ====================================================================
@app.get("/", include_in_schema=False)
def read_root(request: Request): return templates.TemplateResponse("index.html", {"request": request})


@app.get("/live", include_in_schema=False)
def live_page(request: Request): return templates.TemplateResponse("live.html", {"request": request})


@app.get("/search", include_in_schema=False)
def search_page(request: Request): return templates.TemplateResponse("search.html", {"request": request})


@app.get("/recordings", include_in_schema=False)
def recordings_page(request: Request): return templates.TemplateResponse("recordings.html", {"request": request})


@app.get("/playback", include_in_schema=False)
def playback_page(request: Request): return templates.TemplateResponse("playback.html", {"request": request})


# --- Authentication & User Endpoints ---
async def get_current_active_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                          detail="Could not validate credentials",
                                          headers={"WWW-Authenticate": "Bearer"}, )
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_username(db, username=token_data.username)
    if user is None or not user.is_active: raise credentials_exception
    return user


@app.post("/token", response_model=schemas.Token, tags=["Authentication"])
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register/", response_model=schemas.User, tags=["Users"])
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user: raise HTTPException(status_code=400, detail="Username already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/me/", response_model=schemas.User, tags=["Users"])
async def read_users_me(current_user: models.User = Depends(get_current_active_user)): return current_user


# --- Location & ROI Endpoints ---
@app.get("/locations/", response_model=List[schemas.Location], tags=["Locations"])
def read_locations_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)): return crud.get_locations(
    db, skip=skip, limit=limit)


@app.get("/locations/{location_id}", response_model=schemas.Location, tags=["Locations"])
def read_location_endpoint(location_id: int, db: Session = Depends(get_db)):
    db_location = crud.get_location(db, location_id=location_id)
    if db_location is None: raise HTTPException(status_code=404, detail="Location not found")
    return db_location

# === THÊM ENDPOINT MỚI ĐỂ TẠO VỊ TRÍ ===
@app.post("/api/locations/", response_model=schemas.Location, status_code=status.HTTP_201_CREATED, tags=["Locations"])
def create_new_location(
    location: schemas.LocationCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Tạo một vị trí giám sát mới. Yêu cầu đăng nhập.
    Tên vị trí phải là duy nhất.
    """
    db_location = crud.get_location_by_name(db, name=location.name)
    if db_location:
        raise HTTPException(status_code=400, detail="Tên vị trí đã tồn tại.")
    new_location = crud.create_location(db=db, location=location)
    return new_location

@app.put("/api/locations/{location_id}/roi", status_code=status.HTTP_200_OK, tags=["Locations"])
async def update_location_roi(location_id: int, roi_update: schemas.ViolationRoiUpdate,
                              current_user: models.User = Depends(get_current_active_user),
                              db: Session = Depends(get_db)):
    location = crud.get_location(db, location_id)
    if not location: raise HTTPException(status_code=404, detail="Location not found")
    updated_location = crud.update_location_roi(db, location_id, roi_update.violation_roi)
    if stream_manager.current_location_id == location_id:
        stream_manager.current_violation_roi = updated_location.violation_roi
        status_message = {"type": "status_update", "location_id": updated_location.id,
                          "location_name": updated_location.name, "violation_roi": updated_location.violation_roi,
                          "detection_roi": stream_manager.current_detection_roi,
                          "ai_detection_enabled": stream_manager.ai_detection_enabled}
        await stream_manager.broadcast_message(json.dumps(status_message))
    return {"message": "Vùng ROI Vi phạm đã được cập nhật.", "location": updated_location}


@app.put("/api/locations/{location_id}/detection_roi", status_code=status.HTTP_200_OK, tags=["Locations"])
async def update_location_detection_roi_endpoint(location_id: int, roi_update: schemas.ViolationRoiUpdate,
                                                 current_user: models.User = Depends(get_current_active_user),
                                                 db: Session = Depends(get_db)):
    location = crud.get_location(db, location_id)
    if not location: raise HTTPException(status_code=404, detail="Location not found")
    updated_location = crud.update_location_detection_roi(db, location_id, roi_update.violation_roi)
    if stream_manager.current_location_id == location_id:
        stream_manager.current_detection_roi = updated_location.detection_roi
        status_message = {"type": "status_update", "location_id": updated_location.id,
                          "location_name": updated_location.name, "violation_roi": stream_manager.current_violation_roi,
                          "detection_roi": updated_location.detection_roi,
                          "ai_detection_enabled": stream_manager.ai_detection_enabled}
        await stream_manager.broadcast_message(json.dumps(status_message))
    return {"message": "Vùng ROI Nhận diện đã được cập nhật.", "location": updated_location}


# --- Stream Control & WebSocket ---
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await stream_manager.add_viewer(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await stream_manager.remove_viewer(websocket)


@app.put("/api/streams/detection_mode/{mode}", status_code=status.HTTP_200_OK, tags=["Streams"])
async def set_detection_mode(mode: str, current_user: models.User = Depends(get_current_active_user)):
    if mode == "start":
        stream_manager.ai_detection_enabled = True;
        print("[*] Chế độ phát hiện AI đã được BẬT.")
    elif mode == "stop":
        stream_manager.ai_detection_enabled = False;
        print("[*] Chế độ phát hiện AI đã được TẮT.")
    else:
        raise HTTPException(status_code=400, detail="Chế độ không hợp lệ.")
    status_message = {"type": "status_update", "location_id": stream_manager.current_location_id,
                      "location_name": stream_manager.current_location_name,
                      "violation_roi": stream_manager.current_violation_roi,
                      "detection_roi": stream_manager.current_detection_roi,
                      "ai_detection_enabled": stream_manager.ai_detection_enabled}
    await stream_manager.broadcast_message(json.dumps(status_message))
    return {"message": f"Chế độ phát hiện AI đã được chuyển sang: {mode}"}


@app.post("/api/streams/switch/{location_id}", status_code=status.HTTP_202_ACCEPTED, tags=["Streams"])
async def switch_active_stream(location_id: int, current_user: models.User = Depends(get_current_active_user)):
    db = SessionLocal()
    location = crud.get_location(db, location_id)
    db.close()
    if not location: raise HTTPException(status_code=404, detail="Location not found")
    asyncio.create_task(stream_manager.switch_stream(location_id))
    return {"message": "Yêu cầu chuyển đổi luồng đã được chấp nhận."}


# --- Violation & Video Endpoints ---
@app.get("/violations/", response_model=List[schemas.Violation], tags=["Violations"])
def read_violations_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db),
                             current_user: models.User = Depends(get_current_active_user)): return crud.get_violations(
    db, skip=skip, limit=limit)


@app.get("/violations/search/{plate_number}", response_model=List[schemas.Violation], tags=["Violations"])
def search_violations(plate_number: str, db: Session = Depends(get_db),
                      current_user: models.User = Depends(get_current_active_user)):
    if not plate_number: return []
    return crud.get_violations_by_plate(db, plate_number=plate_number)


@app.get("/videos/location/{location_id}", response_model=List[schemas.VideoRecording], tags=["Videos"])
def read_videos_for_location(location_id: int, db: Session = Depends(get_db)): return crud.get_videos_by_location_id(db,
                                                                                                                     location_id=location_id)