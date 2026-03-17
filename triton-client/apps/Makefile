################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
################################################################################

CUDA_VER?=12.8
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-server-app
CC:= g++
TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)
NVDS_VERSION:=8.0
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

SRCS:= $(wildcard *.cpp)
INCS:= $(wildcard *.h)
PKGS:= yaml-cpp gstreamer-rtsp-server-1.0 gstreamer-1.0
OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I./include/includes \
         -I./include \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -I/usr/include/jsoncpp

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS)) -lyaml-cpp

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lnvdsgst_customhelper -lnvds_rest_server -lm \
       -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) -lnvds_rest_server

ifeq ($(TARGET_DEVICE),aarch64)
   LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra/ -lgstnvcustomhelper -ljsoncpp
else
   LIBS+= -L$(LIB_INSTALL_DIR) -lgstnvcustomhelper -ljsoncpp
endif

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)