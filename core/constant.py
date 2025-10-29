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
INTERACTION_LABELS = ["Highly Active", "Active", "Dormant"]

# --- Encoding Mappings ---
# Ordinal features and their correct order
ORDINAL_FEATURES_MAP = {
    "Age Group": AGE_LABELS,
    "Interaction Frequency": INTERACTION_LABELS
}