# app/models.py

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from .database import Base
import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    # Lưu tọa độ ROI dạng chuỗi "x1,y1,x2,y2"
    violation_roi = Column(String)

    # === THÊM 2 DÒNG MỚI ĐỂ LƯU TỌA ĐỘ BẢN ĐỒ ===
    latitude = Column(Float, nullable=True)  # Vĩ độ
    longitude = Column(Float, nullable=True)  # Kinh độ

    # Mối quan hệ tới các vi phạm
    violations = relationship("Violation", back_populates="location")

    # --- DÒNG CẦN THÊM ĐỂ SỬA LỖI ---
    # Mối quan hệ ngược lại tới các video đã ghi
    video_recordings = relationship("VideoRecording", back_populates="location")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    license_plate_info = Column(String, index=True, nullable=True)
    overview_image_path = Column(String)
    vehicle_image_path = Column(String)
    license_plate_image_path = Column(String)

    location_id = Column(Integer, ForeignKey("locations.id"))
    location = relationship("Location", back_populates="violations")


class VideoRecording(Base):
    __tablename__ = "video_recordings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    video_path = Column(String, unique=True)

    location_id = Column(Integer, ForeignKey("locations.id"))
    location = relationship("Location", back_populates="video_recordings")
