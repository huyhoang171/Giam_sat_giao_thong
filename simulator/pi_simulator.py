import cv2
import socket
import struct
import time
import os

# --- CẤU HÌNH ---
SERVER_IP = "127.0.0.1"  
SERVER_PORT = 9999     
VIDEO_PATH = "1009.mp4" 

# Kiểm tra xem file video có tồn tại không
if not os.path.exists(VIDEO_PATH):
    print(f"LỖI: Không tìm thấy file video '{VIDEO_PATH}'.")
    print("Vui lòng đặt một file video vào cùng thư mục với script này và đặt tên là 'input_video.mp4', hoặc thay đổi biến VIDEO_PATH.")
    exit()

def send_video_frames():
    """
    Mở video, đọc từng frame và gửi qua socket đến server.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"LỖI: Không thể mở video file '{VIDEO_PATH}'.")
        return

    print(f"[*] Đang đọc video từ: {VIDEO_PATH}")

    while True:
        client_socket = None
        try:
            print(f"[*] Đang kết nối tới server tại {SERVER_IP}:{SERVER_PORT}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))
            print("[+] Kết nối thành công tới server.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Đã đọc hết video, hoặc có lỗi khi đọc
                    print("[*] Đã đọc hết video. Đang tua lại và bắt đầu gửi lại...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Tua video về đầu
                    ret, frame = cap.read() # Đọc frame đầu tiên sau khi tua lại
                    if not ret:
                        print("[!] Lỗi khi tua video về đầu hoặc video trống. Dừng mô phỏng.")
                        break # Thoát nếu không thể đọc lại

                if frame is None or frame.size == 0:
                    print("[!] Frame trống hoặc không hợp lệ. Bỏ qua.")
                    continue

                # Mã hóa frame thành JPEG
                # Chất lượng nén 80 (có thể điều chỉnh từ 0-100)
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = img_encoded.tobytes()

                # Đóng gói kích thước dữ liệu vào 4 bytes (unsigned long - ">L")
                size = struct.pack(">L", len(data))

                # Gửi kích thước và sau đó là dữ liệu ảnh
                client_socket.sendall(size + data)

                # Giả lập độ trễ của Pi (khoảng 20-30 FPS)
                time.sleep(0.02) # Điều chỉnh để thay đổi tốc độ gửi frame

        except ConnectionRefusedError:
            print("[!] Kết nối bị từ chối. Server có thể chưa chạy hoặc IP/cổng không đúng. Thử lại sau 5 giây...")
            time.sleep(5)
        except ConnectionResetError:
            print("[!] Kết nối bị reset bởi peer. Server có thể đã đóng kết nối. Đang kết nối lại...")
            if client_socket:
                client_socket.close()
            time.sleep(1)
        except BrokenPipeError:
            print("[!] Broken pipe. Kết nối bị mất. Đang kết nối lại...")
            if client_socket:
                client_socket.close()
            time.sleep(1)
        except Exception as e:
            print(f"[!!!] Đã xảy ra lỗi không mong muốn: {e}. Đang kết nối lại...")
            if client_socket:
                client_socket.close()
            time.sleep(2)
        finally:
            if client_socket:
                client_socket.close()
                print("[*] Socket client đã đóng.")

    cap.release()
    print("[*] Đã dừng đọc video.")

if __name__ == "__main__":
    send_video_frames()