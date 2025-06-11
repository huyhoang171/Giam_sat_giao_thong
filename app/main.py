import asyncio
import cv2
import base64
import time
import os
import re  # Thêm import re để sử dụng trong hàm normalize
from unidecode import unidecode  # Thêm import unidecode

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import subprocess
import shutil
from datetime import timedelta, datetime
from typing import List, Dict, Set, Optional
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
STREAM_SOURCE_URL = "tcp://127.0.0.1:5000"

# --- Khởi tạo ---
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Hệ thống Giám sát Giao thông Đơn luồng")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ====================================================================
# === HÀM HELPER ===
# ====================================================================

def normalize_filename(name: str) -> str:
    """
    Chuẩn hóa chuỗi để tạo tên file an toàn:
    - Chuyển tiếng Việt có dấu thành không dấu.
    - Chuyển thành chữ thường.
    - Thay thế khoảng trắng và các ký tự không an toàn bằng gạch dưới.
    """
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
        self.viewers: Set[WebSocket] = set()
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.detector = ViolationDetector()
        self.ai_detection_enabled: bool = False

    async def _capture_loop(self):
        if self.current_location_id is None:
            return

        location_id = self.current_location_id
        stream_url = STREAM_SOURCE_URL

        print(f"[*] Bắt đầu capture cho vị trí {self.current_location_name} (ID: {location_id}) từ nguồn: {stream_url}")
        os.makedirs("static/videos", exist_ok=True)

        VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
        RECORDING_DURATION_SECONDS = 60

        # SỬA LỖI: Chuẩn hóa tên file để an toàn với FFMPEG
        location_name_for_file = normalize_filename(self.current_location_name)

        while self.current_location_id == location_id:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[!] Lỗi: Không thể mở nguồn stream: {stream_url}. Thử lại sau 5s.")
                await asyncio.sleep(5)
                continue

            VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FRAME_RATE = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if VIDEO_WIDTH == 0 or VIDEO_HEIGHT == 0:
                print(f"[!] Lỗi: Kích thước video từ stream không hợp lệ. Thử lại sau 5 giây.")
                cap.release()
                await asyncio.sleep(5)
                continue

            video_writer = None
            last_saved_time = time.time()
            video_filename = ""

            while self.current_location_id == location_id:
                if video_writer is None or (time.time() - last_saved_time) >= RECORDING_DURATION_SECONDS:
                    if video_writer:
                        video_writer.release()
                        print(f"[*] Đã ghi xong file video: {video_filename}")
                        await asyncio.to_thread(save_video_to_db, location_id, video_filename)

                    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    video_filename = f"static/videos/{location_name_for_file}_{ts_str}.mp4"
                    video_writer = cv2.VideoWriter(video_filename, VIDEO_CODEC, FRAME_RATE, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    last_saved_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    print(f"[!] Mất kết nối tới nguồn stream. Tự động kết nối lại.")
                    break

                if video_writer is not None:
                    video_writer.write(frame)

                try:
                    processed_frame = frame.copy()
                    if self.ai_detection_enabled:
                        roi_str = self.current_violation_roi
                        violation_roi = None
                        if roi_str and roi_str != 'None':
                            try:
                                roi_coords = list(map(int, roi_str.split(',')))
                                if len(roi_coords) == 4:
                                    violation_roi = tuple(roi_coords)
                            except ValueError:
                                violation_roi = None

                        if violation_roi:
                            processed_frame, violations_data_list = self.detector.process_frame(frame, violation_roi)
                        else:
                            violations_data_list = []

                        if violations_data_list:
                            for violation_data in violations_data_list:
                                overview_dir = "static/violations/overview"
                                vehicle_dir = "static/violations/vehicle"
                                plate_dir = "static/violations/plate"

                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                unique_filename_prefix = f"{location_name_for_file}_{timestamp_str}"

                                overview_path = save_frame_as_image(violation_data["overview_frame"], overview_dir,
                                                                    f"overview_{unique_filename_prefix}")
                                vehicle_path = save_frame_as_image(violation_data["vehicle_frame"], vehicle_dir,
                                                                   f"vehicle_{unique_filename_prefix}")
                                license_plate_path = save_frame_as_image(violation_data["plate_frame"], plate_dir,
                                                                         f"plate_{unique_filename_prefix}")

                                db_violation = schemas.ViolationCreate(
                                    license_plate_info=violation_data["license_plate_info"],
                                    overview_image_path=overview_path,
                                    vehicle_image_path=vehicle_path,
                                    license_plate_image_path=license_plate_path
                                )
                                db = SessionLocal()
                                try:
                                    crud.create_violation(db, db_violation, location_id)
                                    print(f"[*] Đã lưu vi phạm {violation_data['license_plate_info']} vào CSDL.")
                                except Exception as e:
                                    print(f"[!] Lỗi khi lưu vi phạm vào CSDL: {e}")
                                finally:
                                    db.close()

                    if self.viewers and not self.frame_queue.full():
                        await self.frame_queue.put(processed_frame)

                except Exception as e:
                    print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH XỬ LÝ FRAME: {e}")
                    print(f"[!!!] Bỏ qua frame này để tránh làm sập luồng stream. Vui lòng kiểm tra lỗi trên.\n")
                    if self.viewers and not self.frame_queue.full():
                        await self.frame_queue.put(frame)

                await asyncio.sleep(max(0, 1.0 / FRAME_RATE - 0.005))

            cap.release()
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
                await self.broadcast_message(jpg_as_text)
            except asyncio.TimeoutError:
                continue

    async def broadcast_message(self, message: str):
        disconnected_clients = set()
        for ws in self.viewers:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected_clients.add(ws)
        if disconnected_clients:
            self.viewers -= disconnected_clients

    async def switch_stream(self, new_location_id: int):
        if self.current_location_id == new_location_id:
            return

        if self.current_stream_task:
            self.current_location_id = None
            self.current_stream_task.cancel()
            try:
                await self.current_stream_task
            except asyncio.CancelledError:
                pass

        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()

        db = SessionLocal()
        new_location = crud.get_location(db, new_location_id)
        db.close()

        if not new_location:
            print(f"[!] Lỗi: Vị trí {new_location_id} không tồn tại.")
            self.current_location_name = "Không có"
            self.current_location_id = None
            self.current_violation_roi = None
            self.ai_detection_enabled = False
            return

        self.current_location_id = new_location.id
        self.current_location_name = new_location.name
        self.current_violation_roi = new_location.violation_roi
        self.ai_detection_enabled = False

        capture_task = asyncio.create_task(self._capture_loop())
        broadcast_task = asyncio.create_task(self._broadcast_loop())
        self.current_stream_task = asyncio.gather(capture_task, broadcast_task)

        print(f"[*] Đã chuyển sang giám sát tại: {self.current_location_name} (ID: {self.current_location_id})")

        status_message = {
            "type": "status_update",
            "location_id": self.current_location_id,
            "location_name": self.current_location_name,
            "violation_roi": self.current_violation_roi,
            "ai_detection_enabled": self.ai_detection_enabled
        }
        await self.broadcast_message(json.dumps(status_message))

    async def add_viewer(self, websocket: WebSocket):
        await websocket.accept()
        self.viewers.add(websocket)
        print(f"[*] Một người xem đã kết nối. Tổng số người xem: {len(self.viewers)}")
        status_message = {
            "type": "status_update",
            "location_id": self.current_location_id,
            "location_name": self.current_location_name,
            "violation_roi": self.current_violation_roi,
            "ai_detection_enabled": self.ai_detection_enabled
        }
        await websocket.send_text(json.dumps(status_message))
        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(jpg_as_text)
            except asyncio.QueueEmpty:
                pass

    async def remove_viewer(self, websocket: WebSocket):
        self.viewers.discard(websocket)
        print(f"[*] Một người xem đã ngắt kết nối. Tổng số người xem: {len(self.viewers)}")


stream_manager = StreamManager()


# ====================================================================
# === SỰ KIỆN STARTUP/SHUTDOWN CỦA SERVER ===
# ====================================================================
@app.on_event("startup")
async def startup_event():
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
    print("[*] Đã dừng stream.")


# ====================================================================
# === API ENDPOINTS VÀ WEBSOCKET ===
# ====================================================================
@app.get("/", include_in_schema=False)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/live", include_in_schema=False)
def live_page(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})


@app.get("/search", include_in_schema=False)
def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/recordings", include_in_schema=False)
def recordings_page(request: Request):
    return templates.TemplateResponse("recordings.html", {"request": request})


@app.get("/playback", include_in_schema=False)
def playback_page(request: Request):
    return templates.TemplateResponse("playback.html", {"request": request})


@app.get("/api/locations/{location_id}/roi", response_model=schemas.ViolationRoiResponse, tags=["Locations"])
def get_location_roi(location_id: int, db: Session = Depends(get_db)):
    location = crud.get_location(db, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"location_id": location.id, "violation_roi": location.violation_roi}


async def get_current_active_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
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


@app.put("/api/locations/{location_id}/roi", status_code=status.HTTP_200_OK, tags=["Locations"])
async def update_location_roi(
        location_id: int,
        roi_update: schemas.ViolationRoiUpdate,
        current_user: models.User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    location = crud.get_location(db, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    updated_location = crud.update_location_roi(db, location_id, roi_update.violation_roi)
    stream_manager.current_violation_roi = updated_location.violation_roi
    status_message = {
        "type": "status_update",
        "location_id": updated_location.id,
        "location_name": updated_location.name,
        "violation_roi": updated_location.violation_roi,
        "ai_detection_enabled": stream_manager.ai_detection_enabled
    }
    await stream_manager.broadcast_message(json.dumps(status_message))
    return {"message": "Vùng ROI đã được cập nhật thành công.", "location": updated_location}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await stream_manager.add_viewer(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "request_current_stream_info":
                    status_message = {
                        "type": "status_update",
                        "location_id": stream_manager.current_location_id,
                        "location_name": stream_manager.current_location_name,
                        "violation_roi": stream_manager.current_violation_roi,
                        "ai_detection_enabled": stream_manager.ai_detection_enabled
                    }
                    await websocket.send_text(json.dumps(status_message))
            except json.JSONDecodeError:
                pass  # Bỏ qua các tin nhắn không phải JSON
    except WebSocketDisconnect:
        pass
    finally:
        await stream_manager.remove_viewer(websocket)


@app.put("/api/streams/detection_mode/{mode}", status_code=status.HTTP_200_OK, tags=["Streams"])
async def set_detection_mode(
        mode: str,
        current_user: models.User = Depends(get_current_active_user)
):
    if mode == "start":
        stream_manager.ai_detection_enabled = True
        print("[*] Chế độ phát hiện AI đã được BẬT.")
    elif mode == "stop":
        stream_manager.ai_detection_enabled = False
        print("[*] Chế độ phát hiện AI đã được TẮT.")
    else:
        raise HTTPException(status_code=400, detail="Chế độ không hợp lệ. Chỉ chấp nhận 'start' hoặc 'stop'.")

    status_message = {
        "type": "status_update",
        "location_id": stream_manager.current_location_id,
        "location_name": stream_manager.current_location_name,
        "violation_roi": stream_manager.current_violation_roi,
        "ai_detection_enabled": stream_manager.ai_detection_enabled
    }
    await stream_manager.broadcast_message(json.dumps(status_message))
    return {"message": f"Chế độ phát hiện AI đã được chuyển sang: {mode}"}


@app.post("/api/streams/switch/{location_id}", status_code=status.HTTP_202_ACCEPTED, tags=["Streams"])
async def switch_active_stream(
        location_id: int,
        current_user: models.User = Depends(get_current_active_user)
):
    db = SessionLocal()
    location = crud.get_location(db, location_id)
    db.close()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    asyncio.create_task(stream_manager.switch_stream(location_id))
    return {"message": "Yêu cầu chuyển đổi luồng đã được chấp nhận."}


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
async def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    return current_user


@app.get("/locations/", response_model=List[schemas.Location], tags=["Locations"])
def read_locations_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_locations(db, skip=skip, limit=limit)


@app.get("/locations/{location_id}", response_model=schemas.Location, tags=["Locations"])
def read_location_endpoint(location_id: int, db: Session = Depends(get_db)):
    db_location = crud.get_location(db, location_id=location_id)
    if db_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    return db_location


@app.get("/violations/", response_model=List[schemas.Violation], tags=["Violations"])
def read_violations_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_db),
                             current_user: models.User = Depends(get_current_active_user)):
    return crud.get_violations(db, skip=skip, limit=limit)


@app.get("/violations/search/{plate_number}", response_model=List[schemas.Violation], tags=["Violations"])
def search_violations(
        plate_number: str,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_active_user)
):
    if not plate_number: return []
    return crud.get_violations_by_plate(db, plate_number=plate_number)


@app.get("/videos/location/{location_id}", response_model=List[schemas.VideoRecording], tags=["Videos"])
def read_videos_for_location(location_id: int, db: Session = Depends(get_db)):
    videos = crud.get_videos_by_location_id(db, location_id=location_id)
    return videos