################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Makefile for Dynamic YOLO Ultralytics Parser
# Builds libnvdsinfer_yolo_ultralytics.so
#
# Key Feature: Auto-detects number of classes from tensor shape
################################################################################

# CUDA Version - set based on your DeepStream version
# DeepStream 8.0 = 12.8
# DeepStream 7.1 = 12.6
# DeepStream 7.0/6.4 = 12.2
CUDA_VER?=12.8

# Paths
DEEPSTREAM_PATH?=/opt/nvidia/deepstream/deepstream
CUDA_PATH?=/usr/local/cuda-$(CUDA_VER)

# Compiler settings
CXX:=g++

# Library name
LIB_NAME:=libnvdsinfer_yolo_ultralytics.so

# Source files
SRCS:=nvdsinfer_yolo_ultralytics.cpp
OBJS:=$(SRCS:.cpp=.o)

# Include paths
INCLUDES:= \
	-I$(CUDA_PATH)/include \
	-I$(DEEPSTREAM_PATH)/sources/includes

# Library paths
LIBS:= \
	-L$(CUDA_PATH)/lib64 \
	-L$(DEEPSTREAM_PATH)/lib

# Compiler flags
CXXFLAGS:= -O3 -fPIC -std=c++14 -Wall -Wno-deprecated-declarations
LDFLAGS:= -shared -Wl,--no-undefined

# Optional: Enable debug output
ifdef DEBUG
CXXFLAGS+= -DPARSER_DEBUG=1 -g
endif

# Optional: Custom thresholds
ifdef CONF_THRESH
CXXFLAGS+= -DCONF_THRESHOLD=$(CONF_THRESH)
endif

ifdef NMS_THRESH
CXXFLAGS+= -DNMS_IOU_THRESHOLD=$(NMS_THRESH)
endif

ifdef MAX_DETS
CXXFLAGS+= -DMAX_DETECTIONS=$(MAX_DETS)
endif

# Default target
all: $(LIB_NAME)

$(LIB_NAME): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo ""
	@echo "=============================================="
	@echo " Built: $(LIB_NAME)"
	@echo " CUDA Version: $(CUDA_VER)"
	@echo "=============================================="
	@echo ""
	@echo " DYNAMIC CLASS DETECTION ENABLED"
	@echo " Works with ANY number of classes!"
	@echo ""
	@echo " Available parse functions:"
	@echo "   - NvDsInferParseYoloUltralytics (auto-detect)"
	@echo "   - NvDsInferParseYoloV5"
	@echo "   - NvDsInferParseYoloV8"
	@echo "   - NvDsInferParseYolo11"
	@echo ""

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB_NAME)

install: $(LIB_NAME)
	cp $(LIB_NAME) $(DEEPSTREAM_PATH)/lib/
	@echo "Installed to $(DEEPSTREAM_PATH)/lib/$(LIB_NAME)"

# Debug build
debug: CXXFLAGS+= -DPARSER_DEBUG=1 -g -O0
debug: clean all

# Show exported symbols
symbols: $(LIB_NAME)
	nm -D $(LIB_NAME) | grep -i "nvdsinfer\|yolo"

help:
	@echo "Dynamic YOLO Ultralytics Parser - Build Options"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the library (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install to DeepStream lib directory"
	@echo "  debug    - Build with debug output enabled"
	@echo "  symbols  - Show exported symbols"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_VER=X.Y      - Set CUDA version (default: 12.8)"
	@echo "  DEBUG=1           - Enable debug prints"
	@echo "  CONF_THRESH=0.25  - Set confidence threshold"
	@echo "  NMS_THRESH=0.45   - Set NMS IoU threshold"
	@echo ""

.PHONY: all clean install debug symbols help
