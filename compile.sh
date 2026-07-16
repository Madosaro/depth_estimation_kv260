#!/bin/sh

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

if [ $1 = kv260 ]; then
      ARCH=./arch_$2.json
      TARGET=kv260
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR KV260.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, vck190, u50 ..exiting"
      exit 1
fi

MODEL_NAME=$3
BUILD_DIR=./models/${MODEL_NAME}/build_$2
QUANT_MODEL_PATH=${BUILD_DIR}/quant_model/UNet_int.xmodel

COMPILED_DIR=${BUILD_DIR}/compiled_model

mkdir -p ${COMPILED_DIR}
    

compile() {
  vai_c_xir \
  --xmodel      ${QUANT_MODEL_PATH} \
  --arch        $ARCH \
  --net_name    CNN_${TARGET} \
  --output_dir  ${COMPILED_DIR}
}

compile 2>&1 | tee ${COMPILED_DIR}/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"

