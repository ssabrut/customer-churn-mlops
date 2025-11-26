#!/bin/bash

echo 'Waiting for MinIO...'

sleep 5

mc alias set myminio http://s3:9000 ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}

mc mb --ignore-existing myminio/mlartifacts

mc policy set download myminio/mlartifacts

echo 'Bucket mlartifacts created and policy set.'