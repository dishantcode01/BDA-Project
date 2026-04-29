CREATE DATABASE IF NOT EXISTS earthquake_db;
USE earthquake_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    depth FLOAT NOT NULL,
    magnitude FLOAT NOT NULL,
    risk VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
