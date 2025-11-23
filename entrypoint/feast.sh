#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the feature repository
cd feature_repo

# Apply the feature definitions to the registry/infrastructure
echo "Applying Feast features..."
feast apply

# Navigate back (optional, depending on where your app runs)
cd ..

# Run the main application
# "$@" executes the CMD from the Dockerfile
exec "$@"