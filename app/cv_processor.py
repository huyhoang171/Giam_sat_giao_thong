import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

try:
    from sort.sort import Sort
except ImportError:
    print("[!] CẢNH BÁO: Không tìm thấy thư viện 'sort'.")
    Sort = None

# =====================================================================================
# HẰNG SỐ CẤU HÌNH
# =====================================================================================
VALID_DIRECTION_VECTOR = np.array([0, -1])
DIRECTION_DOT_PRODUCT_THRESHOLD = 0.3
MIN_HISTORY_POINTS = 5
MIN_MOVEMENT_MAGNITUDE = 5


# === CÁC HÀM TIỆN ÍCH ===
def _preprocess_image_white(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def _preprocess_image_black(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


# =======================================================================================
# === HÀM _extract_characters ĐÃ ĐƯỢC SỬA LỖI HOÀN CHỈNH ===
# =======================================================================================
def _extract_characters(thresh_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Trích xuất và sắp xếp các bounding box của ký tự từ ảnh ngưỡng.
    Sắp xếp các ký tự thành hàng từ trên xuống dưới, và trong mỗi hàng từ trái qua phải.
    """
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 5 < w < 50 and 15 < h < 50:
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])

    rows: List[List[Tuple[int, int, int, int]]] = []

    current_row = [boxes[0]]
    for box in boxes[1:]:
        if abs(box[1] - current_row[-1][1]) < 20:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]

    rows.append(sorted(current_row, key=lambda b: b[0]))

    return [box for row in rows for box in row]


# =======================================================================================

def get_traffic_light_state(light_image: Optional[np.ndarray]) -> str:
    if light_image is None or light_image.size == 0: return "UNKNOWN"
    hsv_image = cv2.cvtColor(light_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70]);
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70]);
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100]);
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 70, 70]);
    upper_green = np.array([90, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    pixel_threshold = max(5, int(light_image.shape[0] * light_image.shape[1] * 0.01))
    if red_pixels > pixel_threshold and red_pixels >= yellow_pixels and red_pixels >= green_pixels: return "RED"
    if yellow_pixels > pixel_threshold and yellow_pixels >= red_pixels and yellow_pixels >= green_pixels: return "YELLOW"
    if green_pixels > pixel_threshold and green_pixels >= red_pixels and green_pixels >= yellow_pixels: return "GREEN"
    return "UNKNOWN"


class ViolationDetector:
    def __init__(self,
                 yolo_model_path: str = "../runs/detect/yolo/best.pt",
                 lp_model_path: str = "../runs/detect/license_plate/best_model.h5"):
        print("[*] Đang khởi tạo bộ xử lý CV...")
        try:
            print("[*] Đang tải model YOLO lên GPU...")
            self.yolo_model = YOLO(yolo_model_path)
            self.yolo_model.to('cuda')
            print(f"[*] Model YOLO đã được tải thành công lên GPU.")
            self.class_names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"LỖI: Không thể tải model YOLO lên GPU. Lỗi: {e}")
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[*] TensorFlow đã phát hiện {len(gpus)} GPU(s).")
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("[*] TensorFlow không phát hiện thấy GPU.")
            self.lp_model = load_model(lp_model_path)
            self.lp_labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            print(f"[*] Model đọc biển số đã được tải từ: {lp_model_path}")
        except Exception as e:
            raise RuntimeError(f"LỖI: Không thể tải model đọc biển số. Lỗi: {e}")

        if Sort:
            self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        else:
            self.tracker = None

        self.violation_ids: set[int] = set()
        self.current_light_state: str = "UNKNOWN"
        self.track_history: Dict[int, deque] = {}
        self.vehicle_state: Dict[int, Tuple[int, str]] = {}
        self.early_plate_images: Dict[int, np.ndarray] = {}  # Đổi tên cho rõ ràng hơn

        print("[*] Khởi tạo bộ xử lý CV thành công.")

    def _read_license_plate(self, plate_image_np: Optional[np.ndarray]) -> str:
        try:
            if plate_image_np is None or plate_image_np.size == 0: return "Anh bien so loi"
            thresh_white = _preprocess_image_white(plate_image_np)
            thresh_black = _preprocess_image_black(plate_image_np)

            # Sử dụng hàm _extract_characters đã sửa lỗi
            char_regions = _extract_characters(thresh_white)
            if not char_regions:
                char_regions = _extract_characters(thresh_black)
                if not char_regions: return "Khong tim thay ky tu"

            characters = []
            for x, y, w, h in char_regions:
                # Dùng ảnh thresh_black (nền đen chữ trắng) để predict là tốt nhất cho model MNIST-like
                char_img = thresh_black[y:y + h, x:x + w]
                char_img = cv2.resize(char_img, (28, 28))
                char_img = char_img.astype("float32") / 255.0
                char_img = np.expand_dims(char_img, axis=-1)
                characters.append(char_img)

            if not characters: return "Khong trich xuat duoc"

            predictions = self.lp_model.predict(np.array(characters), verbose=0)
            return ''.join([self.lp_labels[np.argmax(p)] for p in predictions]) or "Rong"
        except Exception as e:
            print(f"[!] Lỗi khi xử lý biển số: {e}")
            return "Loi xu ly"

    def _extract_plate_image_with_fallback(self, frame: np.ndarray, vehicle_box: Tuple[int, int, int, int],
                                           all_plate_boxes: List[List[int]]) -> Optional[np.ndarray]:
        x1_v, y1_v, x2_v, y2_v = vehicle_box
        for lp_box in all_plate_boxes:
            x1_lp, y1_lp, x2_lp, y2_lp = lp_box
            center_lp_x = (x1_lp + x2_lp) / 2
            center_lp_y = (y1_lp + y2_lp) / 2
            if x1_v < center_lp_x < x2_v and y1_v < center_lp_y < y2_v:
                return frame[y1_lp:y2_lp, x1_lp:x2_lp]
        vehicle_img = frame[y1_v:y2_v, x1_v:x2_v]
        if vehicle_img.size > 0:
            h, w = vehicle_img.shape[:2]
            y_start, y_end = int(h * 0.55), int(h * 0.95)
            x_start, x_end = int(w * 0.20), int(w * 0.80)
            return vehicle_img[y_start:y_end, x_start:x_end]
        return None

    def process_frame(self, frame: np.ndarray, violation_roi: Tuple[int, int, int, int]) -> Tuple[
        np.ndarray, List[Dict[str, Any]]]:
        processed_frame = frame.copy()
        violations_in_frame: List[Dict[str, Any]] = []
        roi_x1, roi_y1, roi_x2, roi_y2 = violation_roi

        cv2.rectangle(processed_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
        cv2.putText(processed_frame, "Vung Vuot Den Do", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

        results = self.yolo_model(frame, verbose=False, half=True, imgsz=640)
        detections_for_sort, license_plates_boxes, light_box_img, max_light_conf = [], [], None, 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2, conf, cls_id = *map(int, box.xyxy[0]), float(box.conf[0]), int(box.cls[0])
                label = self.class_names[cls_id]
                if label == "Den hieu" and conf > max_light_conf:
                    max_light_conf, light_box_img = conf, frame[y1:y2, x1:x2]
                elif label == "Bien so":
                    license_plates_boxes.append([x1, y1, x2, y2])
                elif label in ["Xe may", "O to"]:
                    detections_for_sort.append([x1, y1, x2, y2, conf])

        new_light_state = get_traffic_light_state(light_box_img)
        if new_light_state != "UNKNOWN":
            self.current_light_state = new_light_state

        if self.current_light_state == "GREEN":
            self.violation_ids.clear()
            self.vehicle_state.clear()
            self.early_plate_images.clear()

        light_color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}.get(self.current_light_state,
                                                                                              (255, 255, 255))
        cv2.putText(processed_frame, f"Light: {self.current_light_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    light_color, 2, cv2.LINE_AA)

        tracked_objects = self.tracker.update(
            np.array(detections_for_sort)) if self.tracker and detections_for_sort else []

        active_track_ids = set()

        for obj in tracked_objects:
            x1b, y1b, x2b, y2b, track_id = map(int, obj)
            active_track_ids.add(track_id)

            cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=20)
                # Cắt và lưu biển số ngay khi xe mới xuất hiện
                plate_img = self._extract_plate_image_with_fallback(frame, (x1b, y1b, x2b, y2b), license_plates_boxes)
                if plate_img is not None and plate_img.size > 0:
                    self.early_plate_images[track_id] = plate_img

            self.track_history[track_id].append((cx, cy))

            current_zone_state, light_at_entry = self.vehicle_state.get(track_id, (0, ""))
            new_zone_state = current_zone_state

            if current_zone_state == 0:
                if cy <= roi_y2:
                    new_zone_state = 1
                    light_at_entry = self.current_light_state
            elif current_zone_state == 1:
                if cy <= roi_y1:
                    new_zone_state = 2

            self.vehicle_state[track_id] = (new_zone_state, light_at_entry)

            is_in_valid_direction = False
            if len(self.track_history[track_id]) >= MIN_HISTORY_POINTS:
                movement_vector = np.array(self.track_history[track_id][-1]) - np.array(self.track_history[track_id][0])
                if np.linalg.norm(movement_vector) > MIN_MOVEMENT_MAGNITUDE:
                    unit_movement_vector = movement_vector / np.linalg.norm(movement_vector)
                    if np.dot(unit_movement_vector, VALID_DIRECTION_VECTOR) > DIRECTION_DOT_PRODUCT_THRESHOLD:
                        is_in_valid_direction = True

            is_new_violator = track_id not in self.violation_ids
            has_passed_through = (current_zone_state == 1 and new_zone_state == 2)
            is_red_light_at_exit = self.current_light_state == "RED"

            box_color = (255, 150, 0)

            if is_new_violator and has_passed_through and is_red_light_at_exit and is_in_valid_direction:
                print(f"====> VI PHAM XAC NHAN: XE {track_id} <====")
                print(f"  - Ly do: Da hoan thanh viec di xuyen qua vung vi pham luc den do.")

                box_color = (0, 0, 255)
                self.violation_ids.add(track_id)

                vehicle_img_np = frame[y1b:y2b, x1b:x2b]
                plate_img_to_read = self.early_plate_images.get(track_id)

                plate_text = self._read_license_plate(plate_img_to_read)

                violations_in_frame.append({
                    "license_plate_info": plate_text, "overview_frame": frame.copy(),
                    "vehicle_frame": vehicle_img_np, "plate_frame": plate_img_to_read
                })

            elif track_id in self.violation_ids:
                box_color = (0, 165, 255)

            cv2.rectangle(processed_frame, (x1b, y1b), (x2b, y2b), box_color, 2)
            cv2.putText(processed_frame, f"ID:{track_id}", (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Xóa các track_id cũ không còn được theo dõi để giải phóng bộ nhớ
        inactive_ids = set(self.early_plate_images.keys()) - active_track_ids
        for inactive_id in inactive_ids:
            if inactive_id in self.early_plate_images:
                del self.early_plate_images[inactive_id]
            if inactive_id in self.vehicle_state:
                del self.vehicle_state[inactive_id]
            if inactive_id in self.track_history:
                del self.track_history[inactive_id]

        return processed_frame, violations_in_frame