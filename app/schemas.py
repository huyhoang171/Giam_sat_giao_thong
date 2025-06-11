import datetime
from typing import List, Optional

from pydantic import BaseModel

# --- Schemas cho Xác thực (Authentication) ---

class Token(BaseModel):
    """
    Schema cho dữ liệu token trả về khi đăng nhập thành công.
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Schema cho dữ liệu được mã hóa bên trong JWT.
    """
    username: Optional[str] = None


# --- Schemas cho Người dùng (User) ---

class UserBase(BaseModel):
    """
    Schema cơ bản cho user, chỉ chứa username.
    """
    username: str


class UserCreate(UserBase):
    """
    Schema dùng khi tạo user mới, yêu cầu có mật khẩu.
    """
    password: str


class User(UserBase):
    """
    Schema dùng khi đọc/trả về thông tin user từ database.
    Không chứa mật khẩu để đảm bảo an toàn.
    """
    id: int
    is_active: bool

    class Config:
        # Cho phép Pydantic đọc dữ liệu từ các đối tượng ORM (SQLAlchemy models)
        from_attributes = True


# --- Schemas cho Vị trí (Location) ---

class LocationBase(BaseModel):
    """
    Schema cơ bản cho vị trí, chứa các thông tin cần thiết khi tạo.
    """
    name: str
    violation_roi: str  # Ví dụ: "100,300,750,550"

    # === THÊM 2 DÒNG MỚI ===
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LocationCreate(LocationBase):
    """
    Schema dùng khi tạo một vị trí mới (không cần sửa, tự kế thừa).
    """
    pass


class Location(LocationBase):
    """
    Schema dùng khi đọc/trả về thông tin vị trí từ database.
    """
    id: int

    class Config:
        from_attributes = True

# THÊM: Schema mới cho phản hồi ROI
class ViolationRoiResponse(BaseModel):
    location_id: int
    violation_roi: Optional[str] = None

# THÊM: Schema mới để cập nhật ROI
class ViolationRoiUpdate(BaseModel):
    violation_roi: str


# --- Schemas cho Video đã ghi (VideoRecording) ---

class VideoRecordingBase(BaseModel):
    """
    Schema cơ bản cho một bản ghi video.
    """
    video_path: str


class VideoRecordingCreate(VideoRecordingBase):
    """
    Schema dùng khi tạo một bản ghi video mới.
    """
    pass


class VideoRecording(VideoRecordingBase):
    """
    Schema dùng khi đọc/trả về thông tin video từ database.
    Chứa thông tin vị trí được lồng vào.
    """
    id: int
    timestamp: datetime.datetime
    location: Location  # Lồng schema Location vào đây

    class Config:
        from_attributes = True


# --- Schemas cho Vi phạm (Violation) ---

class ViolationBase(BaseModel):
    """
    Schema cơ bản cho một vi phạm, chứa các đường dẫn tới ảnh bằng chứng.
    """
    license_plate_info: Optional[str] = None
    overview_image_path: str
    vehicle_image_path: str
    license_plate_image_path: str


class ViolationCreate(ViolationBase):
    """
    Schema dùng khi tạo một bản ghi vi phạm mới.
    """
    pass


class Violation(ViolationBase):
    """
    Schema dùng khi đọc/trả về thông tin vi phạm từ database.
    Chứa thông tin vị trí được lồng vào.
    """
    id: int
    timestamp: datetime.datetime
    location: Location  # Lồng schema Location vào đây

    class Config:
        from_attributes = True