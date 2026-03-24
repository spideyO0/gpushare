#include <stdio.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;

int main() {
    void *lib = dlopen("/usr/local/lib/gpushare/libgpushare_client.so.1", RTLD_LAZY);
    if (!lib) {
        printf("FAIL: Cannot load library: %s\n", dlerror());
        return 1;
    }
    
    CUresult (*cuInit)(unsigned int) = (CUresult (*)(unsigned int))dlsym(lib, "cuInit");
    CUresult (*cuDeviceGetCount)(int*) = (CUresult (*)(int*))dlsym(lib, "cuDeviceGetCount");
    CUresult (*cuDeviceGet)(CUdevice*, int) = (CUresult (*)(CUdevice*, int))dlsym(lib, "cuDeviceGet");
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = (CUresult (*)(char*, int, CUdevice))dlsym(lib, "cuDeviceGetName");
    
    if (!cuInit) {
        printf("FAIL: cuInit not found\n");
        return 1;
    }
    
    printf("Step 1: cuInit...\n");
    CUresult err = cuInit(0);
    printf("cuInit result: %d\n", err);
    
    printf("Step 2: cuDeviceGetCount...\n");
    int count = 0;
    err = cuDeviceGetCount(&count);
    printf("Device count: %d (err=%d)\n", count, err);
    
    printf("Step 3: cuDeviceGet(0)...\n");
    CUdevice dev;
    err = cuDeviceGet(&dev, 0);
    printf("Device 0 handle: %d (err=%d)\n", dev, err);
    
    printf("Step 4: cuDeviceGetName...\n");
    char name[256];
    err = cuDeviceGetName(name, sizeof(name), dev);
    printf("Device name: %s (err=%d)\n", name, err);
    
    return 0;
}
