# lie_detection    

我们团队查阅了相关文献，在测谎研究的最新进展的启发下，试图利用面部微表情特征和语音特征进行测谎训练，从而开发出一套测谎系统。

首先，我们设计了实验数据集。我们将面部微表情特征分为5个特征点，将声音特征分为22个特征点。为了定位人脸的特征点，我们使用了面部坐标检测的方法；为了分析语音特征，我们使用了librosa团队提供的语音收集分析技术。

其次，为了收集足够多的实验数据，我们设计了数据采集系统以及数据采集实验的流程。实验分为三个步骤，第一个步骤，让志愿者说谎言，获取谎言数据；第二个步骤，让志愿者说实话，获取真相数据；第三个步骤，让志愿者回答开放性的问题，并作记录。准备好录音笔、摄像头、笔记本电脑等实验仪器，再邀请来自四川大学的24位本科生做志愿者，就可以开始收集实验数据了。

然后，搭建面部微表情测谎模型和语音测谎模型，开始训练。经过不断地尝试，不断地分析，不断地调整网络结构，终于训练出了最佳的模型，面部微表情测谎模型的准确率达到了79.8%，语音测谎模型的准确率达到了64.5%。

最后，使用这两个模型成功开发出了实时测谎系统。




