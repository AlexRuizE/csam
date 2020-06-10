FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git

RUN pip install sklearn

WORKDIR /usr/local/

RUN git clone --single-branch --branch psing-dock https://309035ee655387cd13482d335d24b80edecd014c@github.com/AlexRuizE/csam.git

WORKDIR /usr/local/csam

ENTRYPOINT [ "python", "detection/tile_detector.py" ]
