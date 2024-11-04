clear all;
close all;
k=imread('1.png'); %image to be read 
I=k(:,:,1);
faceDetect = vision.CascadeObjectDetector();
bbox=step(faceDetect,I);
face=imcrop(I,bbox);
centerx=size(face,1)/2+bbox(1);
centery=size(face,2)/2+bbox(2);
eyeDetect = vision.CascadeObjectDetector('RightEye');
eyebox=step(eyeDetect,face);
n=size(eyebox,1);
e=[];
for it=1:n
    for j=1:n
        if (j > it)
          if ((abs(eyebox(j,2)-eyebox(it,2))<68)&& (abs(eyebox(j,1)-eyebox(it,1))>40))
            e(1,:)=eyebox(it,:);
            e(2,:)=eyebox(j,:);
            d=1;break;
          end
        end
    end
    if(d == 1)
        break;
    end
end
eyebox(1,:)=e(1,:);
eyebox(2,:)=e(2,:);
c=eyebox(1,3)/2;
d=eyebox(1,4)/2;
eyeCenter1x=eyebox(1,1)+c+bbox(1);
eyeCenter1y=eyebox(1,2)+d+bbox(2);
e=eyebox(2,3)/2;
f=eyebox(2,4)/2;
eyeCenter2x=eyebox(2,1)+e+bbox(1);
eyeCenter2y=eyebox(2,2)+f+bbox(2);
ndetect=vision.CascadeObjectDetector('Nose','MergeThreshold',16);
nosebox=step(ndetect,face);
noseCenterx=nosebox(1,1)+(nosebox(1,3)/2)+bbox(1);
noseCentery=nosebox(1,2)+(nosebox(1,4)/2);
m=[1,noseCentery,size(face,1),((size(face,2))-noseCentery)];
mouth=imcrop(face,m);

mdetect=vision.CascadeObjectDetector('Mouth','MergeThreshold' ,20);
mouthbox=step(mdetect,mouth);
for it=1:size(mouthbox,1)
    if(mouthbox(it,2)>20)
        mouthbox(1,:)=mouthbox(it,:);
        break;
    end
end
mouthbox(1,2)=mouthbox(1,2)+noseCentery;
noseCentery=noseCentery+bbox(2);
%ind=find(mb(:,1)==max(mb(:,1)));
%mb=mb(ind,:);
mouthCenterx=mouthbox(1,1)+(mouthbox(1,3)/2)+bbox(1);
mouthCentery=mouthbox(1,2)+(mouthbox(1,4)/2)+bbox(2);
shape=[centerx centery;eyeCenter1x eyeCenter1y;eyeCenter2x eyeCenter2y;noseCenterx noseCentery;mouthCenterx mouthCentery];
imshow(I);hold on;plot(shape(:,1),shape(:,2),'+','MarkerSize',10);
%To get the bound boxes as shown in the figure uncomment the below code and 
%comment the imshow(I);hold on;plot(shape(:,1),shape(:,2),'+',... line
% eyebox(1,1:2)=eyebox(1,1:2)+bbox(1,1:2);
% eyebox(2,1:2)=eyebox(2,1:2)+bbox(1,1:2);
% nosebox(1,1:2)=nosebox(1,1:2)+bbox(1,1:2);
% mouthbox(1,1:2)=mouthbox(1,1:2)+bbox(1,1:2);
% all_points=[eyebox(1,:);eyebox(2,:);nosebox(1,:);mouthbox(1,:)];
% dpoints=size(all_points,1);
% label=cell(dpoints,1);
% i=1;
% for i = 1: dpoints
% label{i}=[num2str(i)];
% i=i+1;
% end
% videoout=insertObjectAnnotation(I,'rectangle',all_points,label,'TextBoxOpacity',0.3,'Fontsize',9);
% imshow(videoout);hold on;plot(shape(:,1),shape(:,2),'+','MarkerSize',10);
%To get triangles uncomment the below code
%dt=DelaunayTri(shape(:,1),shape(:,2));
%imshow(videoout);hold on;triplot(dt);hold off