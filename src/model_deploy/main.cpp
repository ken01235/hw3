#include <math.h>
#include "mbed.h"
#include "mbed_rpc.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "uLCD_4DGL.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "stm32l475e_iot01_accelero.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
InterruptIn btn(USER_BUTTON);
BufferedSerial pc(USBTX, USBRX);
void gestureUI(Arguments *in, Reply *out);
void tiltUI(Arguments *in, Reply *out);
RPCFunction rpcGuesture(&gestureUI, "gesture");
RPCFunction rpcTilt(&tiltUI, "tilt");
void selectfreq(void);
int  PredictGesture(float*);
void selectfreq_terminate();
int THangle = 180;
int terminate1 = 0;
int terminate2 = 0;
void angleDetect(void);
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
uLCD_4DGL uLCD(D1, D0, D2);
Thread t;
Thread t2;
WiFiInterface *wifi;
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;
const char* topic = "Mbed";
Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;
void messageArrived(MQTT::MessageData& md);
void publish_message(int, float);
void close_mqtt();
MQTT::Client<MQTTNetwork, Countdown> *client;

int main(void) {

    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
        printf("ERROR: No WiFiInterface found.\r\n");
        return -1;
    }

    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
        printf("\nConnection error: %d\r\n", ret);
        return -1;
    }

    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client1(mqttNetwork);
    client = &client1;

    //TODO: revise host to your IP
    const char* host = "192.168.43.177";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
        printf("Connection error.");
        return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client1.connect(data)) != 0){
        printf("Fail to connect MQTT\r\n");
    }
    if (client1.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
        printf("Fail to subscribe\r\n");
    }

    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    btn.rise(mqtt_queue.event(&selectfreq_terminate));
    //btn3.rise(&close_mqtt);

    char buf[256], outbuf[256];

    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");

    while(1) {
        memset(buf, 0, 256);
        for (int i = 0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }
}

void gestureUI(Arguments *in, Reply *out) {
    int first = 1;
    myled1 = 1;
    terminate1 = 0;
    if(first)
        t.start(selectfreq);
    return;
}

void tiltUI(Arguments *in, Reply *out) {
    int first = 1;
    myled2 = 1;
    terminate2 = 0;
    if (first)
        t2.start(angleDetect);
    return;
}

int PredictGesture(float* output) {
    static int continuous_count = 0;
    static int last_predict = -1;

    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict) {
        continuous_count += 1;
    } else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}

void selectfreq(void) {
    uLCD.cls();
    uLCD.text_width(2);
    uLCD.text_height(2);
    uLCD.printf("selection:\n");

    while(1){
        bool should_clear_buffer = false;
        bool got_data = false;

        int gesture_index;

        static tflite::MicroErrorReporter micro_error_reporter;
        tflite::ErrorReporter* error_reporter = &micro_error_reporter;

        const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            error_reporter->Report(
                                   "Model provided is schema version %d not equal "
                                   "to supported version %d.",
                                   model->version(), TFLITE_SCHEMA_VERSION);
            return;
        }

        static tflite::MicroOpResolver<6> micro_op_resolver;
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                     tflite::ops::micro::Register_MAX_POOL_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                     tflite::ops::micro::Register_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                     tflite::ops::micro::Register_FULLY_CONNECTED());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                     tflite::ops::micro::Register_SOFTMAX());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                     tflite::ops::micro::Register_RESHAPE(), 1);

        static tflite::MicroInterpreter static_interpreter(
                                                           model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
        tflite::MicroInterpreter* interpreter = &static_interpreter;

        interpreter->AllocateTensors();

        TfLiteTensor* model_input = interpreter->input(0);
        if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
            (model_input->dims->data[1] != config.seq_length) ||
            (model_input->dims->data[2] != kChannelNumber) ||
            (model_input->type != kTfLiteFloat32)) {
            error_reporter->Report("Bad input tensor parameters in model");
            return;
        }

        int input_length = model_input->bytes / sizeof(float);

        TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
        if (setup_status != kTfLiteOk) {
            error_reporter->Report("Set up failed\n");
            return;
        }

        error_reporter->Report("Set up successful...\n");

        while (true) {

            got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                         input_length, should_clear_buffer);

            if (!got_data) {
                should_clear_buffer = false;
                continue;
            }

            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                error_reporter->Report("Invoke failed on index: %d\n", begin_index);
                continue;
            }

            gesture_index = PredictGesture(interpreter->output(0)->data.f);

            should_clear_buffer = gesture_index < label_num;

            if (gesture_index < label_num) {
                //error_reporter->Report(config.output_message[gesture_index]);
                THangle = (gesture_index + 1) * 45;
                uLCD.cls();
                uLCD.text_width(2);
                uLCD.text_height(2);
                uLCD.printf("selection:\n");
                uLCD.text_width(3);
                uLCD.text_height(3);
                uLCD.printf("%d\n", THangle);
            }
            while (terminate1) {
                ThisThread::sleep_for(100ms);
            }
        }
    }
}

void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    ++arrivedcount;
}

void publish_message(int mode, float data) {
    if (mode == 1 && terminate1) return;
    message_num++;
    MQTT::Message message;
    char buff[100];
    if (mode == 1)
        sprintf(buff, "The threshold angle is: %d\n", (int)data);
    else if (mode == 2)
        sprintf(buff, "The last angle is: %10.6lf\n", data);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);

    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);

}

void close_mqtt() {
    closed = true;
}

void selectfreq_terminate() {
    publish_message(1, (float)THangle);
    terminate1 = 1;
    myled1 = 0;
}

void angleDetect() {
    int16_t XYZ[3] = {0};
    BSP_ACCELERO_Init();
    while (true){
        BSP_ACCELERO_AccGetXYZ(XYZ);
        float a = sqrt(XYZ[0] * XYZ[0] + XYZ[1] * XYZ[1]);
        float b = XYZ[2];
        float angle = atan(a / b) * 180 / 3.1415926;
        if (angle < 0.0)
            angle = angle + 180.0;
        else if (angle > 180.0)
            angle = angle - 180.0;
        //printf("angle: %10.6lf\n", angle);
        if (angle >= THangle) {
            terminate2 = 1;
        }
        if (terminate2) {
            publish_message(2, angle);
            myled2 = 0;
            while (terminate2) {
                ThisThread::sleep_for(100ms);
            }
        }
        ThisThread::sleep_for(100ms);
    }
}
