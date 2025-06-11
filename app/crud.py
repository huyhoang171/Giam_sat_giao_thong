from sqlalchemy.orm import Session, joinedload
from . import models, schemas, auth


# --- User CRUD: Các hàm xử lý người dùng ---

def get_user(db: Session, user_id: int):
    """
    Lấy thông tin một người dùng bằng ID.
    """
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    """
    Lấy thông tin một người dùng bằng tên đăng nhập (username).
    """
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    """
    Lấy danh sách nhiều người dùng, có phân trang.
    """
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    """
    Tạo một người dùng mới. Mật khẩu sẽ được băm trước khi lưu.
    """
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# --- Location CRUD: Các hàm xử lý vị trí giám sát ---

def get_location(db: Session, location_id: int):
    """
    Lấy thông tin một vị trí bằng ID.
    """
    return db.query(models.Location).filter(models.Location.id == location_id).first()


def get_location_by_name(db: Session, name: str):
    """
    Lấy thông tin một vị trí bằng tên.
    """
    return db.query(models.Location).filter(models.Location.name == name).first()


def get_locations(db: Session, skip: int = 0, limit: int = 100):
    """
    Lấy danh sách các vị trí, có phân trang.
    """
    return db.query(models.Location).offset(skip).limit(limit).all()


def create_location(db: Session, location: schemas.LocationCreate):
    """
    Tạo một vị trí giám sát mới, bao gồm cả tọa độ.
    """
    db_location = models.Location(
        name=location.name,
        violation_roi=location.violation_roi,
        latitude=location.latitude,
        longitude=location.longitude,
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

# THÊM: Hàm cập nhật ROI cho vị trí
def update_location_roi(db: Session, location_id: int, new_roi: str):
    """
    Cập nhật vùng ROI cho một vị trí cụ thể.
    """
    db_location = db.query(models.Location).filter(models.Location.id == location_id).first()
    if db_location:
        db_location.violation_roi = new_roi
        db.commit()
        db.refresh(db_location)
    return db_location


# --- Violation CRUD: Các hàm xử lý các trường hợp vi phạm ---

def get_violations(db: Session, skip: int = 0, limit: int = 20):
    """
    Lấy danh sách các vi phạm gần đây, có phân trang.
    Sử dụng joinedload để tải kèm thông tin vị trí trong cùng một câu query,
    giúp tối ưu hiệu năng, tránh N+1 query problem.
    """
    return db.query(models.Violation).options(
        joinedload(models.Violation.location)
    ).order_by(models.Violation.timestamp.desc()).offset(skip).limit(limit).all()


def create_violation(db: Session, violation: schemas.ViolationCreate, location_id: int):
    """
    Tạo một bản ghi vi phạm mới, liên kết với một vị trí cụ thể.
    """
    db_violation = models.Violation(
        license_plate_info=violation.license_plate_info,
        overview_image_path=violation.overview_image_path,
        vehicle_image_path=violation.vehicle_image_path,
        license_plate_image_path=violation.license_plate_image_path,
        location_id=location_id
    )
    db.add(db_violation)
    db.commit()
    db.refresh(db_violation)
    return db_violation


# --- VideoRecording CRUD: Các hàm xử lý video đã ghi ---

def get_video_recordings(db: Session, skip: int = 0, limit: int = 20):
    """
    Lấy danh sách các video đã được ghi gần đây, có phân trang.
    Cũng sử dụng joinedload để tải kèm thông tin vị trí.
    """
    return db.query(models.VideoRecording).options(
        joinedload(models.VideoRecording.location)
    ).order_by(models.VideoRecording.timestamp.desc()).offset(skip).limit(limit).all()


def create_video_recording(db: Session, video: schemas.VideoRecordingCreate, location_id: int):
    """
    Tạo một bản ghi video mới, liên kết với một vị trí.
    """
    db_video = models.VideoRecording(
        video_path=video.video_path,
        location_id=location_id
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def get_violations_by_plate(db: Session, plate_number: str):
    """
    Lấy danh sách các vi phạm dựa trên biển số xe.
    Sử dụng 'ilike' để tìm kiếm không phân biệt chữ hoa, chữ thường.
    """
    search_pattern = f"%{plate_number}%"
    return db.query(models.Violation).options(
        joinedload(models.Violation.location)
    ).filter(
        models.Violation.license_plate_info.ilike(search_pattern)
    ).order_by(models.Violation.timestamp.desc()).all()

def get_videos_by_location_id(db: Session, location_id: int, skip: int = 0, limit: int = 100):
    """Lấy danh sách các video đã ghi của một vị trí cụ thể."""
    return db.query(models.VideoRecording).filter(
        models.VideoRecording.location_id == location_id
    ).order_by(models.VideoRecording.timestamp.desc()).offset(skip).limit(limit).all()