#include "mbed.h"
#include "mbed_rpc.h"
#include "stm32l475e_iot01_accelero.h"

RpcDigitalOut myled1(LED1,"myled1");
RpcDigitalOut myled2(LED1,"myled2");
BufferedSerial pc(USBTX, USBRX);
RPCFunction rpcAcc(&getAcc, "getAcc");
void getAcc(Arguments *in, Reply *out);
void gestureUI(Arguments *in, Reply *out);
void tiltAngle(Arguments *in, Reply *out);

int main() {
    // Enable the the accelerometer
    printf("Start accelerometer init\n");
    BSP_ACCELERO_Init();

    char buf[256], outbuf[256];

    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");

    while (true) {
        memset(buf, 0, 256);      // clear buffer
        for(int i=0; i<255; i++) {
            char recv = fgetc(devin);
            if (recv == '\r' || recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }
}

void getAcc(Arguments *in, Reply *out) {
    int16_t pDataXYZ[3] = {0};
    char buffer[200];
    BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    sprintf(buffer, "Accelerometer values: (%d, %d, %d)", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
    out->putData(buffer);
}

void gestureUI(Arguments *in, Reply *out)
{
    myled1 = 1;

}

void tiltAngle(Arguments *in, Reply *out)
{

}

