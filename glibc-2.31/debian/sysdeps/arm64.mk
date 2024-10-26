# configuration options for all flavours
extra_config_options = --enable-multi-arch

ifeq (,$(filter stage1 stage2, $(DEB_BUILD_PROFILES)))
# build an LSE optimized library
GLIBC_PASSES += lse
GLIBC_PASSES_NOCHECK += lse
DEB_ARCH_REGULAR_PACKAGES += libc6-lse
lse_CC = $(CC) -march=armv8-a+lse
lse_CXX = $(CXX) -march=armv8-a+lse
lse_slibdir = /lib/$(DEB_HOST_MULTIARCH)/atomics
endif
