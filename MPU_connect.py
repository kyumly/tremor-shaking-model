import serial
import time
import torch
from trains.models import CustomModel
import torch.nn.functional as F

py_serial = serial.Serial(

    # Window
    port='COM3',

    # 보드 레이트 (통신 속도)
    baudrate=115200,
)

model = CustomModel()
model.load_state_dict(torch.load('./trains/save.pt'))



data_list = []

while True:
    # commend = input('아두이노에게 내릴 명령:')
    #
    # py_serial.write(commend.encode())
    #
    # time.sleep(0.1)
    if py_serial.readable():
        # 들어온 값이 있으면 값을 한 줄 읽음 (BYTE 단위로 받은 상태)
        # BYTE 단위로 받은 response 모습 : b'\xec\x97\x86\xec\x9d\x8c\r\n'
        response = py_serial.readline()
        print(response)
        # 디코딩 후, 출력 (가장 끝의 \n을 없애주기위해 슬라이싱 사용)
        data = response[:len(response) - 1].decode()
        data = data.replace("end\r", "")

        values = data.split(', ')

        try:

            print(values)
            Axyz = [float(value) for value in values[0:3]]
            Gxyz = [float(value) for value in values[3:6]]
            # Mxyz = [float(value) for value in values[6:7]]

            combined_list = []
            combined_list.extend(Axyz)
            combined_list.extend(Gxyz)
            # combined_list.extend(Mxyz)

            data = (torch.tensor(combined_list, dtype=torch.float64) )
            print("실행 결과 : ", data)

            output_data = model(data)
            result = F.sigmoid(output_data)
            mask = (result >= torch.FloatTensor([0.5]))
            if mask:
                print("실행ㅉ")
                c = "1"
                c = c.encode('utf-8')
                print(py_serial.write(c))
            else:
                print("거짓")
                c = "0"
                c = c.encode('utf-8')
                print(py_serial.write(c))
        except Exception as e:

            print(e)



