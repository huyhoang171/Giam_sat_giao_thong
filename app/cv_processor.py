# -*- coding: utf-8 -*-
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
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("[!] CẢNH BÁO: Không tìm thấy thư viện 'deep-sort-realtime'.")
    print("[!] Vui lòng cài đặt bằng lệnh: pip install deep-sort-realtime")
    DeepSort = None

# =====================================================================================
# HẰNG SỐ CẤU HÌNH
# =====================================================================================
VALID_DIRECTION_VECTORS = [
    np.array([0, -1]),
    np.array([1, -1]) / np.linalg.norm([1, -1]),
    np.array([-1, -1]) / np.linalg.norm([-1, -1]),
    np.array([1, 0]),
    np.array([-1, 0])
]
DIRECTION_DOT_PRODUCT_THRESHOLD = 0.6
MIN_HISTORY_POINTS = 5
MIN_MOVEMENT_MAGNITUDE = 5


# =======================================================================================
# === CÁC HÀM TIỆN ÍCH (Giữ nguyên không đổi) ===
# =======================================================================================
def improve_image_quality(gray_image, upscale_factor=4):
    h, w = gray_image.shape
    if h == 0 or w == 0: return gray_image
    new_w, new_h = int(w * upscale_factor), int(h * upscale_factor)
    upscaled_image = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(upscaled_image)
    final_image = cv2.medianBlur(clahe_image, 3)
    return final_image


def create_binary_image(enhanced_gray_image):
    _, thresh_inv = cv2.threshold(enhanced_gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_black = cv2.bitwise_not(thresh_inv)
    return thresh_inv, thresh_black


def isolate_plate_area(repaired_thresh_black):
    contours, _ = cv2.findContours(repaired_thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(repaired_thresh_black)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    height, width = repaired_thresh_black.shape
    plate_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(plate_mask, [hull], -1, 255, thickness=cv2.FILLED)
    isolated_result = cv2.bitwise_and(repaired_thresh_black, repaired_thresh_black, mask=plate_mask)
    return isolated_result


def extract_and_sort_all(thresh_img):
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    img_h, img_w = thresh_img.shape[:2]
    min_char_h, max_char_h = img_h * 0.2, img_h * 0.5
    max_char_w, min_area = img_w * 0.4, img_h * img_w * 0.005
    initial_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        if (min_char_h < h < max_char_h) and (w < max_char_w) and (0.1 < aspect_ratio < 2.5) and (
                cv2.contourArea(cnt) > min_area):
            initial_boxes.append((x, y, w, h))
    if not initial_boxes: return []
    final_boxes = []
    for box1 in initial_boxes:
        is_nested = False
        for box2 in initial_boxes:
            if box1 == box2: continue
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            if x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2):
                is_nested = True
                break
        if not is_nested: final_boxes.append(box1)
    if not final_boxes: return []
    rows = []
    threshold = img_h * 0.2
    final_boxes.sort(key=lambda box: box[1])
    for box in final_boxes:
        placed = False
        for row in rows:
            avg_y = sum([b[1] for b in row]) / len(row)
            if abs(avg_y - box[1]) < threshold:
                row.append(box)
                placed = True
                break
        if not placed: rows.append([box])
    rows.sort(key=lambda r: sum([b[1] for b in r]) / len(r))
    sorted_boxes = [box for row in rows for box in sorted(row, key=lambda b: b[0])]
    return sorted_boxes


def post_process_prediction(predictions, labels_char):
    corrected_labels = []
    num_chars = len(predictions)
    for i, pred_vector in enumerate(predictions):
        top3_indices = np.argsort(pred_vector)[:-4:-1]
        top3_labels = [labels_char[idx] for idx in top3_indices]
        final_label = top3_labels[0]
        is_letter_position = (num_chars >= 8 and i == 2)
        if is_letter_position:
            for label in top3_labels:
                if label.isalpha():
                    final_label = label
                    break
        else:
            for label in top3_labels:
                if label.isdigit():
                    final_label = label
                    break
        corrected_labels.append(final_label)
    return corrected_labels


def get_traffic_light_state(light_image: Optional[np.ndarray]) -> str:
    if light_image is None or light_image.size == 0: return "UNKNOWN"
    hsv_image = cv2.cvtColor(light_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 70, 70])
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


# =======================================================================================
# === LỚP XỬ LÝ VI PHẠM (ĐÃ CẬP NHẬT HOÀN CHỈNH) ===
# =======================================================================================

class ViolationDetector:
    def __init__(self,
                 yolo_model_path: str = "runs/detect/yolo/best.pt",
                 lp_model_path: str = "runs/detect/license_plate/best_model_v4.h5",
                 max_plate_images: int = 3):
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
            self.lp_model = load_model(lp_model_path, compile=False)
            self.lp_labels = '0123456789ABCDEFGHKLMNPRSTUVWXYZ'
            print(f"[*] Model đọc biển số đã được tải từ: {lp_model_path}")
        except Exception as e:
            raise RuntimeError(f"LỖI: Không thể tải model đọc biển số. Lỗi: {e}")

        if DeepSort:
            self.tracker = DeepSort(
                max_age=30, n_init=3, nms_max_overlap=0.7, max_cosine_distance=0.4,
                nn_budget=None, embedder="mobilenet", half=True, bgr=True, embedder_gpu=True
            )
            print("[*] Bộ theo dõi DeepSORT đã được khởi tạo trên GPU.")
        else:
            self.tracker = None
            print("[!] LỖI: Không thể khởi tạo DeepSORT.")

        self.CONF_THRESHOLD = 0.4
        self.MAX_PLATE_IMAGES = max_plate_images

        # Gọi hàm reset để khởi tạo trạng thái ban đầu
        self.reset()
        print("[*] Khởi tạo bộ xử lý CV thành công.")

    # === HÀM MỚI ĐỂ RESET TRẠNG THÁI ===
    def reset(self):
        """Xóa sạch trạng thái của bộ phát hiện để chuẩn bị cho luồng mới."""
        print("[*] Resetting ViolationDetector state.")
        self.violation_ids: set[int] = set()
        self.counted_track_ids: set[int] = set()
        self.current_light_state: str = "UNKNOWN"
        self.track_history: Dict[int, deque] = {}
        self.vehicle_state: Dict[int, Tuple[int, str]] = {}
        self.early_plate_images: Dict[int, deque] = {}

    def _read_license_plate(self, plate_image_np: Optional[np.ndarray]) -> str:
        try:
            if plate_image_np is None or plate_image_np.size == 0: return "Anh bien so loi"
            image_gray = cv2.cvtColor(plate_image_np, cv2.COLOR_BGR2GRAY)
            enhanced_gray = improve_image_quality(image_gray)
            _, thresh_black_orig = create_binary_image(enhanced_gray)
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            repaired_thresh_black = cv2.morphologyEx(thresh_black_orig, cv2.MORPH_CLOSE, close_kernel)
            isolated_plate_img = isolate_plate_area(repaired_thresh_black)

            char_boxes = extract_and_sort_all(isolated_plate_img)

            if not char_boxes: return "Khong tim thay ky tu"
            characters_nn = []
            for x, y, w, h in char_boxes:
                char_img_crop = thresh_black_orig[y:y + h, x:x + w]
                padding = 4
                padded_char = cv2.copyMakeBorder(char_img_crop, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                                 value=255)
                resized_char = cv2.resize(padded_char, (28, 28), interpolation=cv2.INTER_AREA)
                processed_char_nn = np.expand_dims(resized_char.astype("float32") / 255.0, axis=-1)
                characters_nn.append(processed_char_nn)
            if not characters_nn: return "Khong trich xuat duoc"
            predictions = self.lp_model.predict(np.array(characters_nn), verbose=0)
            corrected_labels = post_process_prediction(predictions, self.lp_labels)
            return ''.join(corrected_labels) or "Rong"
        except Exception as e:
            print(f"[!] Lỗi khi xử lý biển số: {e}")
            return "Loi xu ly"

    def _extract_plate_image(self, frame: np.ndarray, vehicle_box: Tuple[int, int, int, int],
                             all_plate_boxes: List[List[int]]) -> Optional[np.ndarray]:
        x1_v, y1_v, x2_v, y2_v = vehicle_box
        for lp_box in all_plate_boxes:
            x1_lp, y1_lp, x2_lp, y2_lp = lp_box
            center_lp_x = (x1_lp + x2_lp) / 2
            center_lp_y = (y1_lp + y2_lp) / 2
            if x1_v < center_lp_x < x2_v and y1_v < center_lp_y < y2_v:
                return frame[y1_lp:y2_lp, x1_lp:x2_lp]
        return None

    def process_frame(self, frame: np.ndarray,
                      violation_roi: Tuple[int, int, int, int],
                      detection_roi: Optional[Tuple[int, int, int, int]]
                      ) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        processed_frame = frame.copy()
        violations_in_frame: List[Dict[str, Any]] = []
        newly_confirmed_vehicles = 0

        roi_x1, roi_y1, roi_x2, roi_y2 = violation_roi
        cv2.rectangle(processed_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
        cv2.putText(processed_frame, "Vung Vi Pham", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)

        if detection_roi:
            det_x1, det_y1, det_x2, det_y2 = detection_roi
            cv2.rectangle(processed_frame, (det_x1, det_y1), (det_x2, det_y2), (255, 150, 0), 2)
            cv2.putText(processed_frame, "Vung Nhan Dien", (det_x1, det_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 150, 0), 2)

        results = self.yolo_model(frame, verbose=False, half=True, imgsz=640)
        detections_for_deepsort = []
        license_plates_boxes, light_box_img, max_light_conf = [], None, 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2, conf, cls_id = *map(int, box.xyxy[0]), float(box.conf[0]), int(box.cls[0])
                label = self.class_names[cls_id]
                if label == "Den hieu" and conf > max_light_conf:
                    max_light_conf, light_box_img = conf, frame[y1:y2, x1:x2]
                elif label == "Bien so":
                    license_plates_boxes.append([x1, y1, x2, y2])
                elif label in ["Xe may", "O to"] and conf > self.CONF_THRESHOLD:
                    is_in_detection_roi = True
                    if detection_roi:
                        det_x1, det_y1, det_x2, det_y2 = detection_roi
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        if not (det_x1 < center_x < det_x2 and det_y1 < center_y < det_y2):
                            is_in_detection_roi = False
                    if is_in_detection_roi:
                        w, h = x2 - x1, y2 - y1
                        detections_for_deepsort.append(([x1, y1, w, h], conf, label))

        new_light_state = get_traffic_light_state(light_box_img)
        if new_light_state != "UNKNOWN": self.current_light_state = new_light_state

        if self.current_light_state == "GREEN":
            # Khi đèn xanh, chỉ reset trạng thái liên quan đến vi phạm
            self.violation_ids.clear()
            self.vehicle_state.clear()
            self.early_plate_images.clear()
            # DÒNG GÂY LỖI ĐÃ BỊ XÓA: self.counted_track_ids.clear()

        light_color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}.get(self.current_light_state,
                                                                                              (255, 255, 255))
        cv2.putText(processed_frame, f"Light: {self.current_light_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    light_color, 2, cv2.LINE_AA)

        tracked_objects = self.tracker.update_tracks(detections_for_deepsort,
                                                     frame=frame) if self.tracker and detections_for_deepsort else []
        active_track_ids = set()

        for track in tracked_objects:
            if not track.is_confirmed(): continue
            track_id = int(track.track_id)
            active_track_ids.add(track_id)

            if track_id not in self.counted_track_ids:
                newly_confirmed_vehicles += 1
                self.counted_track_ids.add(track_id)

            ltrb = track.to_tlbr()
            x1b, y1b, x2b, y2b = map(int, ltrb)
            cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2

            if track_id not in self.track_history: self.track_history[track_id] = deque(maxlen=30)
            if track_id not in self.early_plate_images: self.early_plate_images[track_id] = deque(
                maxlen=self.MAX_PLATE_IMAGES)

            if len(self.early_plate_images[track_id]) < self.MAX_PLATE_IMAGES:
                plate_img = self._extract_plate_image(frame, (x1b, y1b, x2b, y2b), license_plates_boxes)
                if plate_img is not None and plate_img.size > 0:
                    self.early_plate_images[track_id].append(plate_img)

            self.track_history[track_id].append((cx, cy))

            current_zone_state, light_at_entry = self.vehicle_state.get(track_id, (0, ""))
            new_zone_state = current_zone_state
            is_inside_horizontally_violation_roi = roi_x1 < cx < roi_x2

            if current_zone_state == 0 and cy <= roi_y2 and is_inside_horizontally_violation_roi:
                new_zone_state = 1
                light_at_entry = self.current_light_state
            elif current_zone_state == 1 and (cy <= roi_y1 or not is_inside_horizontally_violation_roi):
                new_zone_state = 2

            self.vehicle_state[track_id] = (new_zone_state, light_at_entry)

            is_in_valid_direction = False
            if len(self.track_history[track_id]) >= MIN_HISTORY_POINTS:
                movement_vector = np.array(self.track_history[track_id][-1]) - np.array(self.track_history[track_id][0])
                movement_magnitude = np.linalg.norm(movement_vector)
                if movement_magnitude > MIN_MOVEMENT_MAGNITUDE:
                    unit_movement_vector = movement_vector / movement_magnitude
                    for valid_vector in VALID_DIRECTION_VECTORS:
                        if np.dot(unit_movement_vector, valid_vector) > DIRECTION_DOT_PRODUCT_THRESHOLD:
                            is_in_valid_direction = True
                            break

            is_new_violator = track_id not in self.violation_ids
            has_passed_through = (current_zone_state == 1 and new_zone_state == 2)
            is_red_light_at_entry = light_at_entry == "RED"
            box_color = (255, 150, 0)

            if is_new_violator and has_passed_through and is_red_light_at_entry and is_in_valid_direction:
                print(f"====> VI PHAM XAC NHAN: XE {track_id} <====")
                box_color = (0, 0, 255)
                self.violation_ids.add(track_id)
                vehicle_img_np = frame[y1b:y2b, x1b:x2b]
                vehicle_type_detected = track.get_det_class() if track else "Không xác định"

                saved_plates = self.early_plate_images.get(track_id, [])
                plate_text = "Khong doc duoc"
                final_plate_image = None
                for plate_image in reversed(list(saved_plates)):
                    plate_image_for_reading = plate_image
                    try:
                        if plate_image.ndim == 2:
                            plate_image_for_reading = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
                        elif plate_image.shape[2] == 4:
                            plate_image_for_reading = cv2.cvtColor(plate_image, cv2.COLOR_RGBA2BGR)
                    except Exception as e:
                        print(f"[!] Cảnh báo: Lỗi khi xử lý ảnh biển số cho xe {track_id}: {e}")

                    if plate_image_for_reading is not None and plate_image_for_reading.size > 0:
                        recognized_text = self._read_license_plate(plate_image_for_reading)
                        if 7 <= len(
                                recognized_text) <= 10 and "Loi" not in recognized_text and "Khong" not in recognized_text:
                            plate_text = recognized_text
                            final_plate_image = plate_image
                            break
                if final_plate_image is None and saved_plates: final_plate_image = saved_plates[-1]

                violations_in_frame.append({
                    "license_plate_info": plate_text,
                    "vehicle_type": vehicle_type_detected,
                    "overview_frame": frame.copy(),
                    "vehicle_frame": vehicle_img_np,
                    "plate_frame": final_plate_image
                })

            elif track_id in self.violation_ids:
                box_color = (0, 165, 255)

            cv2.rectangle(processed_frame, (x1b, y1b), (x2b, y2b), box_color, 2)
            cv2.putText(processed_frame, f"ID:{track_id}", (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        inactive_ids = set(self.track_history.keys()) - active_track_ids
        for inactive_id in inactive_ids:
            if inactive_id in self.early_plate_images: del self.early_plate_images[inactive_id]
            if inactive_id in self.vehicle_state: del self.vehicle_state[inactive_id]
            if inactive_id in self.track_history: del self.track_history[inactive_id]
            self.counted_track_ids.discard(inactive_id)

        return processed_frame, violations_in_frame, newly_confirmed_vehicles