FROM ubuntu:22.04

WORKDIR /home/

ARG USER_ID
ARG GROUP_ID
ARG VSCODE_COMMIT

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

RUN apt-get -y update && \
    	apt-get -y upgrade && \
	    apt-get -y install git vim wget jq cmake ninja-build clang lld python3-dev python3-distutils software-properties-common libfmt-dev libomp-dev libopenblas-dev protobuf-compiler    

RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 && \
	update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100 && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3 100

RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install tqdm numpy pandas matplotlib ipywidgets notebook timm
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
	groupadd -g ${GROUP_ID} user && \
	useradd -l -u ${USER_ID} -g ${GROUP_ID} user && \
    mkdir /home/user/ ;fi

RUN chown user:user /home/user/

USER user

RUN  if [ $VSCODE_COMMIT ]; then \
	cd /home/user/ && \
	wget https://update.code.visualstudio.com/commit:${VSCODE_COMMIT}/server-linux-x64/stable && \
	tar -xf stable && \
	mkdir -p .vscode-server/bin && \
	mv vscode-server-linux-x64 .vscode-server/bin/${VSCODE_COMMIT} && \
	rm stable ;fi
