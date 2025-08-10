import numpy as np

def Encoder(data):
    data = np.append(data, [0,0,0])  # shift register 설정 및 초기화. 데이터 뒤에 000 추가
    dataSize = np.shape(data)[0]  # 0번째 데이터만 보겠다는 뜻.
    shiftReg = [0, 0, 0]   # k=3일 때 시프트레지스터 초기 설정.
    encoded_bit = np.zeros((2, dataSize))  # R = 1/2  1비트 들어오면 2비트 출력. 2행 dataSize 열
    # 전체다 0으로 만듦. 1비트 들어오면 2비트 출력. 2행 dataSize열만큼 만들어짐

    for i in range(dataSize):
        shiftReg[2] = shiftReg[1]
        shiftReg[1] = shiftReg[0]
        shiftReg[0] = data[i]
        encoded_bit[0, i] = np.logical_xor(np.logical_xor(shiftReg[0], shiftReg[1]), shiftReg[2])
        encoded_bit[1, i] = np.logical_xor(shiftReg[0], shiftReg[2])

    return encoded_bit


def ViterbiDecoder(decoded_bit):
    ref_out = np.zeros((2, 8))
    # 인코딩된 비트 00 / 01 / 10 / 11  들어오는 화살표의 출력!!!들
    ref_out[0, :] = [0, 1, 1, 0, 1, 0, 0, 1]
    ref_out[1, :] = [0, 1, 0, 1, 1, 0, 1, 0]
    # 00으로 들어오는 것의 과거 state는 00과 01
    # 11으로 들어오는 것의 과거 state는 10과 11

    dataSize = np.shape(decoded_bit)[1]  # 2 by 원래데이터길이 + 3 [0,0,0]
    cumDist = [0, 100, 100, 100]   # 초기값 설정 각각 00/01/10/11  state
            #  00  01   10   11
    prevState = []

    decoded_bit = np.array(decoded_bit)  # decoded_bit을 numpy 배열로 변환

    for i in range(dataSize):
        tmpData = np.tile(decoded_bit[:, i].reshape(2, 1), (1, 8))  # 2행 1열로 변환 후 1행 8열로 확장
        dist = np.sum(np.abs(tmpData - ref_out), axis=0)  # 세로 방향으로 더함
        tmpDist = np.tile(cumDist, (1, 2)) + dist
        tmpPrevState = []  # 과거 state 기록
        for a in range(4):   # a값에 0 1 2 3 state 수가 4니까
            if tmpDist[0, 2 * a + 0] <= tmpDist[0, 2 * a + 1]:
                cumDist[a] = tmpDist[0, 2 * a + 0]
                tmpPrevState.append((a % 2) * 2 + 0)
            else:
                cumDist[a] = tmpDist[0, 2 * a + 1]
                tmpPrevState.append((a % 2) * 2 + 1)
        prevState.append(tmpPrevState)

        state_index = np.argmin(cumDist)

    decoded_bit = []

    for b in range(dataSize - 1, -1, -1):  # 디코딩 과정은 역순
        decoded_bit.append(int(state_index / 2))
        state_index = prevState[b][state_index]

    data_size = np.shape(decoded_bit)[0]
    decoded_bit = np.flip(decoded_bit)[0:data_size - 3]  # 마지막 3 비트 제거

    return decoded_bit

