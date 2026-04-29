#!/bin/bash

# EC2 Deployment Script for Earthquake Project
# Usage: Run this on your EC2 instance (Ubuntu recommended)

# 1. Update and install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# 2. Clone the repository (if not already there)
if [ ! -d "earthquake_project" ]; then
    git clone https://github.com/dishantcode01/BDA-Project.git earthquake_project
fi
cd earthquake_project

# 3. Build and run the container
sudo docker build -t earthquake-app .
sudo docker run -d -p 80:5000 \
    --name earthquake-web \
    -e MYSQL_HOST=${MYSQL_HOST:-localhost} \
    -e MYSQL_USER=${MYSQL_USER:-root} \
    -e MYSQL_PASSWORD=${MYSQL_PASSWORD} \
    -e MYSQL_DB=${MYSQL_DB:-earthquake_db} \
    earthquake-app

echo "Deployment complete! Your app should be available at your EC2 Public IP."
