from sqlalchemy import Column, Integer, String, Float, DateTime, Index, UniqueConstraint
from datetime import datetime, timezone
from database import Base

class UserStockTarget(Base):
    __tablename__ = "user_stock_targets"
    #changed: add a composite unique key and lookup index for the hot (user_id, ticker) access pattern
    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_user_stock_targets_user_ticker"),
        Index("ix_user_stock_targets_user_ticker", "user_id", "ticker"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    ticker = Column(String, index=True)
    target_price = Column(Float)
    
    updated_at = Column(
    DateTime, 
    default=lambda: datetime.now(timezone.utc),
    onupdate=lambda: datetime.now(timezone.utc)
)
