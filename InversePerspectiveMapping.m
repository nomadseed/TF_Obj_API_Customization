% Inverse Perspective Mapping for Viewnyx dataset

% given the intrinsic and extrinsic matrices of camera, get an IPM
% note that this is only a demo showing how IPM works

% camera name:Sony IMX111PQ Exmor R
% camera detail:
% https://www.devicespecifications.com/en/model/73c42796
% which provides detailed specifications for the Nexus 4 camera
% pixel width /focal length is 0.00156.  Plugging in the numbers from 
% this site(3.51mm focal length, 3.67 x 2.76 mm sensor size for image with
% 640x480 pixels) gives 0.00163, so the numbers sound reasonable.

% author@wenwen

%% initialize the parameters of camera
alpha=0; % pitch angle alpha, look down if alpha>0
beta=0; % yaw angle beta, look left if beta>0
gamma=0; % roll angle gamma, clockwise if gamma>0
dx=0; % distance from camera to x,move right if dx>0
dy=0; % distance from camera to y,move downward if dy>0
dz=500; % distance from camera to z,step away from frame if dz>0 

w=640; % img width in pixel
h=480; % img height in pixel

focal=0.00351; % focal length in meters, if dz=focal-1, 
% image has 100% solution for image plane is also at z=1
s=5.7344e-06; % sx and sy in intrinsic matrix (in meters), 
% assume all units on the sensor are square, which gives sx=sy=s

%% calculate matrices
% projection mat from 2D to 3D
A=[     1       0       -w/2;
        0       1       -h/2;
        0       0       1;
        0       0       1];

% change angle to rad    
a=(alpha)*pi/180;
b=(beta)*pi/180;
g=(gamma)*pi/180;

% rotation mat x
Rx=[    1      0       0       0;
        0      cos(a)  -sin(a) 0;
        0      sin(a)  cos(a)  0;
        0      0       0       1];

% rotation mat y   
Ry=[    cos(b)  0       sin(b) 0;
        0       1       0       0;
        -sin(b) 0       cos(b)  0;
        0       0       0       1];
    
% rotation mat z
Rz=[    cos(g)  -sin(g) 0       0;
        sin(g)  cos(g)  0       0;
        0       0       1       0;
        0       0       0       1];

% whole rotation mat
R=Rx*Ry*Rz;
    
% translation mat
T=[ 1       0       0       dx;
    0       1       0       dy;
    0       0       1       dz;
    0       0       0       1];

% Intrinsic mat
I=[ focal/s 0       w/2     0;
    0       focal/s h/2     0;
    0       0       1       0];

% complete transformation
M=I*(R*T*A);
% M=[ -0.0653     -1.3908     363.8516;
%     0.1058      -2.2435     562.2521;
%     2.1540e-04  -4.3104e-03 1.0];

%% calculate IPM of an image (without interpolation)
img=imread('4.jpg'); % original image
img_ipm=uint8(zeros(h,w,3));% image of IPM
for y=1:h
   for x=1:w
        pos_new=M*[x;y;1]; % map the [x,y,1] to [t*x',t*y',t]
        pos_new=pos_new./pos_new(3); % get homogeneous coordinates
        pos_new=uint16(floor(pos_new));
        % draw the pixels located in the new img only
        if pos_new(1)>=1 && pos_new(1)<=w && pos_new(2)>=1 && pos_new(2)<=h
            img_ipm(pos_new(2),pos_new(1),:)=img(y,x,:);
        end
   end
end

%% calculate IPM of an image (with interpolation)
img_itp=uint8(zeros(h,w,3));% image of interpolation
Minv=inv(M); % note that Minv*[x;y;1] is no better than M\[x;y;1] in matlab
for y=1:h
    for x=1:w
        pos_bp=M\[x;y;1]; % map the [x',y',1] to [t*x,t*y,t]
        pos_bp=pos_bp./pos_bp(3); % get homogeneous coordinates
        if pos_bp(1)>=2 && pos_bp(1)<=w-1 && pos_bp(2)>=2 && pos_bp(2)<=h-1
            % METHOD 1: copy the value of corresponding integer pixel
            img_itp(y,x,:)=img(round(pos_bp(2)),round(pos_bp(1)),:);
            
            % METHOD 2: take a weighted average of 2-neighbor
            % calculate L1 distance between float pixel coordinate and
            % actual pixel, note that Lx, Ly in [0,1)
%             Lx=pos_bp(1)-floor(pos_bp(1));
%             Ly=pos_bp(1)-floor(pos_bp(1));
%             img_itp(y,x,:)=(Lx*img(ceil(pos_bp(2)),ceil(pos_bp(1)),:)+(1-Lx)*img(ceil(pos_bp(2)),floor(pos_bp(1)),:)+Ly*img(ceil(pos_bp(2)),ceil(pos_bp(1)),:)+(1-Ly)*img(floor(pos_bp(2)),ceil(pos_bp(1)),:))/2;
        end   
    end
end


%% show the img
figure(1);
subplot(2,2,1);
imshow(img);
title('Original image');

subplot(2,2,2);
imshow(img_ipm);
title('IPM without interpolation');

subplot(2,2,3);
imshow(img_itp);
title('IPM with interpolation');

% End of File %