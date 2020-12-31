from tvn.data import SomethingSomethingV2


if __name__ == '__main__':
    data = SomethingSomethingV2(root='./data')
    video, label = data[0]
    print(video.shape)
    print(label)
