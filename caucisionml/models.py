from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, BYTEA, UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import inflection
import uuid

from .database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user_id = Column(String)
    data_imported = Column(Boolean)
    model_trained = Column(Boolean)
    control_promotion = Column(String)
    causal_graph = Column(String)
    data_schema = Column(JSONB)
    graph_order = Column(ARRAY(String))
    promotions = Column(ARRAY(String))
    model = Column(BYTEA)
    model_type = Column(String)

    def data_id(self):
        return f"p_{inflection.underscore(str(self.id))}_data"

    def campaign_data_id(self):
        return f"c_{inflection.underscore(str(self.id))}_data"


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user_id = Column(String)
    project_id = Column(String)
    data_imported = Column(Boolean)
    default = Column(JSONB)
    graph_order = Column(ARRAY(String))
    optimization_result = Column(BYTEA)
    optimization_metadata = Column(BYTEA)

    def data_id(self):
        return f"c_{inflection.underscore(str(self.id))}_data"
