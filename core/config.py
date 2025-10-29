from pydantic_settings import BaseSettings, SettingsConfigDict

class DefaultSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, env_nested_delimiter="__"
    )


class Settings(DefaultSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "churn-api"

def get_settings() -> Settings:
    return Settings()

FEATURES_TO_USE = [
    "Age",
    "Gender",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

TARGET = "Churn"

# --- Feature Engineering Bins ---
# For "Age"
AGE_BINS = [17, 24, 39, 59, 100]  # Bins: (17-24], (24-39], (39-59], (59-100]
AGE_LABELS = ["Young Adult", "Adult", "Mid-Career", "Senior"]

# For "Last Interaction"
INTERACTION_BINS = [-1, 7, 15, 30] # Bins: (0-7], (7-15], (15-30]
INTERACTION_LABELS = ["Highly Active", "Active", "At Risk", "Dormant"]

# --- Encoding Mappings ---
# Ordinal features and their correct order
ORDINAL_FEATURES_MAP = {
    "Age_Binned": AGE_LABELS,
    "Interaction_Binned": INTERACTION_LABELS
}