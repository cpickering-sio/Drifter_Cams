##Creating April Tag PDF with Kalbir
#Initially to run on mac you need to use docker because it needs a linux operating system 

#Download Docker 
#create docker file to initialize the image using this script: 

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget \
    python3-pyx python3-numpy python3-matplotlib

RUN git clone --recursive https://github.com/ethz-asl/kalibr.git /kalibr

WORKDIR /kalibr

# No need to build all of Kalibr for PDF generation
ENV PATH="/kalibr/kalibr/python:${PATH}"

ENTRYPOINT ["/bin/bash"]


#Establish paths (personal to your local files, but here is mine for example)
 docker run -it -v /Users/corinnepickering/Desktop/Thode_lab/stereo-vision/Kalibr:/data kalibr-pdf 

#Establish that you want the pdf to save in the base file with your dockerfile by establishing directory within the environment 
 cd /data

#Create the calibration target 
# nx __ & ny __ changes dimension's while --tsize changes  tag size (meters) and --tspace tag spacing fraction
 python3 /kalibr/aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf --type apriltag --nx 13 --ny 9 --tsize 0.107 --tspace 0.2