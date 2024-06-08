#!/bin/sh
colmap automatic_reconstructor --image_path ${1%/}/images/ --workspace_path ${1%/} --dense 0
colmap image_undistorter --image_path ${1%/}/images/ --input_path ${1%/}/sparse/0 --output_path ${1%/}/undistort --output_type COLMAP
mkdir -p ${1%/}/undistort/sparse/0/
mv ${1%/}/undistort/sparse/* ${1%/}/undistort/sparse/0/ 2>/dev/null
rm -r ${1%/}/sparse/
mv ${1%/}/undistort/sparse/ ${1%/}/sparse/
rm -r ${1%/}/undistort/
