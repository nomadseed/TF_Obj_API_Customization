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
dz=0; % distance from camera to z,step away from frame if dz>0 

w=640; % img width in pixel
h=480; % img height in pixel

focal=1; % focal length

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
I=[ focal   0       w/2     0;
    0       focal   h/2     0;
    0       0       1       0];

% complete transformation
M=I*(R*T*A);
% M=[ -0.0653     -1.3908     363.8516;
%     0.1058      -2.2435     562.2521;
%     2.1540e-04  -4.3104e-03 1.0];

%% calculate IPM of an image
img=imread('4.jpg'); % original image
img_ipm=uint8(zeros(h,w,3));% image of IPM
img_itp=uint8(zeros(h,w,3));% image of interpolation
for y=1:h
   for x=1:w
        % transform [t*x,t*y,t]' into [x,y,1]'
        pos_new=M*[x;y;1];
        pos_new=pos_new./pos_new(3); % get homogeneous coordinates
        pos_new=uint16(floor(pos_new));
        % draw the pixels located in the new img only
        if pos_new(1)>=1 && pos_new(1)<=w && pos_new(2)>=1 && pos_new(2)<=h
            img_ipm(pos_new(2),pos_new(1),:)=img(y,x,:);
        end
   end
end

%% interpolation after transformation


%% show the img
figure(1);
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
imshow(img_ipm);
% subplot(2,2,3);
% imshow(img_itp);

% End of File %