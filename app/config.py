from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LABOR_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

                                 
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    weights_path: Path = Field(default=Path("training/artifacts/weights/best.onnx"))
    zones_config_path: Path = Field(default=Path("training/configs/zones.yaml"))
    sqlite_path: Path = Field(default=Path("data/labor_intel.db"))

                                     
                                                                              
                                                          
    inference_provider: Literal["cuda", "cpu", "auto"] = "auto"
    inference_imgsz: int = 640
    detection_conf_threshold: float = 0.35
    detection_iou_threshold: float = 0.5

                                                                                 
                                                                 
    class_names: tuple[str, ...] = (
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
    )

                                   
                                                                              
                                                                    
    stream_target_fps: int = 3
    stream_queue_maxsize: int = 10
    stream_reconnect_max_seconds: int = 30

                                                
                                     
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8

                                                                                
                                                                             
                                                                            
                                                                       
    activity_velocity_threshold_px: float = 2.0
    activity_velocity_window_frames: int = 5

                                                
    state_active_to_transitioning_sec: int = 5
    state_transitioning_to_idle_sec: int = 15

                                           
                                                                           
                                                                              
    ppe_hardhat_zone: tuple[float, float] = (0.0, 0.33)                           
    ppe_vest_zone: tuple[float, float] = (0.25, 0.75)                  
    ppe_min_iou: float = 0.20

                                     
    analytics_tick_seconds: int = 3
    analytics_low_confidence_min_workers: int = 3

                               
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

                                             
    sqlite_retention_hours: int = 24
    sqlite_retention_check_on_startup: bool = True

                                   
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("weights_path", "zones_config_path", "sqlite_path")
    @classmethod
    def _resolve_relative_paths(cls, v: Path) -> Path:
                                                                         
        return v

    def resolve(self, p: Path) -> Path:
        """Resolve a config path against the project root if it's relative."""
        return p if p.is_absolute() else (self.project_root / p).resolve()

                                                 
settings = Settings()
