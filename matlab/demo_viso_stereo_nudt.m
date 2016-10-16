% demonstrates biologically inspired visual odometry based on the computational model 
% of grid cells on an image sequence
disp('===========================');
clear all; close all; dbstop error;

%parameters setting in the computational model of grid cells proposed in reference to Burk Y, Fiete I (2009) Accurate Path Integration in Continuous Attractor Network Models of Grid Cells. PLoS Computational Biology 5: e1000291

% Timestep in ms - If you are loading your own data, you must change dt to
% your recording time step.
dt = 0.5;%should be adjusted
%----Warning------

% The following are the parameters used in the associated paper. Altering them will
% likely lead to an unsucessful simulation.

% Number of neurons
n = 2^7; %2^7

% Neuron time-constant (in ms)
tau = 5;

% Envelope and Weight Matrix Parameters
lambda = 18; % Equation (3)
beta = 3/lambda^2; % Equation (3)
alphabar = 1.05; % alphabar = gamma/beta from Equation (3)
abar = 1; % a should be <= alphabar^2. Equation (3)
wtphase = 2; % wtphase is 'l' from Equation (2)
alpha  = 2.0;% The velocity gain from Equation (4) 

%parameters setting in the computational model of grid cells

%----------------------
% INITIALIZE VARIABLES
%----------------------

% padding for convolutions
big = 2*n; 
dim = n/2; 

% initial population activity
r=zeros(n,n);  
rfield = r; 
s = r;

% A placeholder for a single neuron response
% sNeuronResponse = zeros(1,sampling_length)';
sNeuron = [n/2, n/2];

% Envelope and Weight Matrix parameters
x = -n/2:1:n/2-1; 
lx=length(x);
xbar=sqrt(beta)*x; 

%------------------------------------
% INITIALIZE SYNAPTIC WEIGHT MATRICES
%------------------------------------

% The idea is to view the population activity as an input signal - x(t), 
% with the weight matrix - h(t), as an impulse response yeilding the 
% output signal as the new population activity - y(t). To get y(t) we will 
% convolute x(t) with h(t).


% The center surround, locally inhibitory weight matrix - Equation (3)

filt = abar*exp(-alphabar*(ones(lx,1)*xbar.^2+xbar'.^2*ones(1,lx)))...
       -exp(-1*(ones(lx,1)*xbar.^2+xbar'.^2*ones(1,lx)));  %acquiring n*n matrix

% The envelope function that determines the global feedforward
% input - Equation (5)

venvelope = exp(-4*(x'.^2*ones(1,n)+ones(n,1)*x.^2)/(n/2)^2); 

% We create shifted weight matrices for each preferred firing direction and
% transform them to obtain h(t).


frshift=circshift(filt,[0,wtphase]);   %right shift
flshift=circshift(filt,[0,-wtphase]);  %left shift
fdshift=circshift(filt,[wtphase,0]);   %down shift
fushift=circshift(filt,[-wtphase,0]);  %up shift

ftu=fft2(fushift,big,big); 
ftd=fft2(fdshift,big,big); 
ftl=fft2(flshift,big,big); 
ftr=fft2(frshift,big,big);

ftu_small=fft2(fftshift(fushift)); 
ftd_small=fft2(fftshift(fdshift)); 
ftl_small=fft2(fftshift(flshift)); 
ftr_small=fft2(fftshift(frshift)); 


% Block matricies used for identifying all neurons of one preferred firing
% direction

typeL=repmat([[1,0];[0,0]],dim,dim);  
typeR=repmat([[0,0];[0,1]],dim,dim);
typeU=repmat([[0,1];[0,0]],dim,dim);  
typeD=repmat([[0,0];[1,0]],dim,dim);  
 
%----------------------------
% INITIAL MOVEMENT CONDITIONS
%----------------------------

theta_v=pi/5;
left = -sin(theta_v); 
right = sin(theta_v);
up = -cos(theta_v); 
down = cos(theta_v); 
vel=0; 

%------------------
% BEGIN SIMULATION 
%------------------
fig = figure(1);
set(fig, 'Position',[50,1000,550,450]);
	

% We run the simulation for 300 ms with aperiodic boundries and 
% zero velocity to form the network, then we change the 
% envelope function to uniform input and continue the 
% simulation with periodic boundry conditions
for iter=1 : 1000
    %----------------------------------------
    % COMPUTE NEURAL POPULATION ACTIVITY 
    %----------------------------------------
    if iter == 800
        venvelope=ones(n,n); 
    end

    % Break global input into its directional components
    % Equation (4)
    rfield = venvelope.*((1+vel*right)*typeR+(1+vel*left)*typeL+(1+vel*up)*typeU+(1+vel*down)*typeD);    
        
    % Convolute population activity with shifted semmetric weights.
    % real() is implemented for octave compatibility
    convolution = real(ifft2(...
			           fft2(r.*typeR,big,big).*ftr ...
			           + fft2(r.*typeL,big,big).*ftl ...
			           + fft2(r.*typeD,big,big).*ftd ...
			           + fft2(r.*typeU,big,big).*ftu));         
     
    % Add feedforward inputs to the shifted population activity to
    % yield the new population activity.
    rfield = rfield+convolution(n/2+1:big-n/2,n/2+1:big-n/2);  

    % Neural Transfer Function
    fr=(rfield>0).*rfield;
      
    % Neuron dynamics - Equation (1)
    r_old = r;
    r_new = min(10,(dt/tau)*(5*fr-r_old)+r_old);
    r = r_new;
        
    %Update the plot every 20 timesteps
    if mod(iter,20)==1
        subplot(2,2,1);
        imagesc(r_new,[0,2]); colormap(hot); colorbar; drawnow;
        title('Neural Population Activity');
    end
end

s = r;
set(fig,'Position',[50,1000,450,900]);
position_x_integrated(1) = 0;
position_y_integrated(1) = 0;

% parameter settings
% the setting may be different if running with different dataset, and it
% depends on different cameras used
img_dir     = '/home/lu/Biologically Inspired Visual Navigation/nudt_dataset';
param.f     = 493.86945;
param.cu    = 320.33392;
param.cv    = 244.18816;
param.base  = 0.11996;

first_frame = 0;
last_frame  = 204;%it depends on how many frames there are in the dataset

% initialize visual odometry
visualOdometryStereoMex('init',param);

% init transformation matrix array
Tr_total{1} = eye(4);

subplot(2,2,1);
imagesc(r_new,[0,2]); colormap(hot); colorbar; drawnow;
title('Neural Population Activity');

% for all frames do
orientation_robot=0;% the robot's orientation

for frame=first_frame:last_frame
  
  % 1-index
  k = frame-first_frame+1;
  
  % read current images
  I1 = rgb2gray(imread([img_dir '/' num2str(frame,'%d') 'L.ppm']));
  I2 = rgb2gray(imread([img_dir '/' num2str(frame,'%d') 'R.ppm']));
  % compute the egomotion
  Tr = visualOdometryStereoMex('process',I1,I2);
  if k>1
    Tr_total{k} = Tr_total{k-1}*inv(Tr);
  end
  Tt=inv(Tr);
  delta_orientation=asin(-Tt(1,3));% the change of the robot's orientation estimated by LIBVISO2
  vel_x_relative=Tt(1,4);% the relative motion in x coordinate estimated by LIBVISO2
  vel_y_relative=Tt(3,4);% the relative motion in y coordinate estimated by LIBVISO2
  if k>1
      tic;
      orientation_robot=orientation_robot+delta_orientation;
      if orientation_robot>pi
          orientation_robot=orientation_robot-2*pi;
      else
          if orientation_robot<-pi
              orientation_robot=orientation_robot+2*pi;
          end
      end
      % the motion information in the world coordinate
      vel_x_world= vel_x_relative*cos(orientation_robot)-vel_y_relative*sin(orientation_robot);
      vel_y_world= vel_x_relative*sin(orientation_robot)+vel_y_relative*cos(orientation_robot);
      % the orientation of velocity vector in the world coordinate
      theta_v =  atan2(vel_y_world, vel_x_world);
      vel = sqrt((vel_x_world)^2 + (vel_y_world)^2);
      left = -cos(theta_v); 
      right = cos(theta_v);
      up = sin(theta_v);
      down = -sin(theta_v);
                 

      % Break feedforward input into its directional components
      % Equation (4) 
      
      for cycle=1:1:50  
          rfield = venvelope.*((1+alpha*vel*right)*typeR+(1+alpha*vel*left)*typeL+(1+alpha*vel*up)*typeU+(1+alpha*vel*down)*typeD);
          % Convolute population activity with shifted semmetric weights.
          % real() is implemented for octave compatibility
          convolution = real(ifft2( ...
                          fft2(r.*typeR).*ftr_small ...
                        + fft2(r.*typeL).*ftl_small ...
                        + fft2(r.*typeD).*ftd_small ... 
                        + fft2(r.*typeU).*ftu_small));  

          % Add feedforward inputs to the shifted population activity to
          % yield the new population activity.

          rfield = rfield+convolution; 

          % Neural Transfer Function
          fr=(rfield>0).*rfield;

          % Neuron dynamics (Eq. 1)
          r_old = r;
          r_new = min(10,(dt/tau)*(5*fr-r_old)+r_old);
          r = r_new;
      end
      
      %recover the robot's displacement from two subsequent frames of neural activations
      interval=1;
      if k==2
          r_old_fordisplacement=r_new;
          fr_old=fr;
          num_integrated=1;
      else if mod(k,interval)==0
              mindis=+inf;
              for shift_x=-9:1:9        
                  for shift_y=-9:1:9    
                      tempfr=circshift(fr_old,[shift_y,shift_x]);
                      tempdis=norm(tempfr-fr);
                      if(tempdis<mindis)
                          mindis=tempdis;
                          displacement_x=shift_x;
                          displacement_y=shift_y;
                      end
                  end
              end

              displace=sqrt((displacement_x)^2 + (displacement_y)^2);
              num_integrated=num_integrated+1;
              % performing the path integration
              if displace~=0
                  position_x_integrated(num_integrated)=position_x_integrated(num_integrated-1)+displacement_x*vel/displace;
                  position_y_integrated(num_integrated)=position_y_integrated(num_integrated-1)-displacement_y*vel/displace;
              else
                  position_x_integrated(num_integrated)=position_x_integrated(num_integrated-1);
                  position_y_integrated(num_integrated)=position_y_integrated(num_integrated-1);
              end
              r_old_fordisplacement=r_new;
              fr_old=fr;
          end
      end
      toc
      
      subplot(2,2,3)
      plot(position_x_integrated(1:num_integrated),position_y_integrated(1:num_integrated),'-',position_x_integrated(num_integrated),position_y_integrated(num_integrated),'o');
      axis([min(position_x_integrated)-5,max(position_x_integrated)+5,min(position_y_integrated)-5,max(position_y_integrated)+5]);
      axis equal;
      drawnow;
      title('Biologically inspired visual odometry');
      xlabel('x: m');
      ylabel('y: m');  
      
      subplot(2,2,1)
      imagesc(r_new,[0,2]); colormap(hot); colorbar; drawnow;
      title('Neural Population Activity');
          
  end
  % update image
  subplot(2,2,2);
  subimage(I1);
  axis image
  title('The left image');

  subplot(2,2,4);
  if k>1
    plot([Tr_total{k-1}(1,4) Tr_total{k}(1,4)], [Tr_total{k-1}(3,4) Tr_total{k}(3,4)],'-xb','LineWidth',2);
    hold on;
    plot(position_x_integrated(1:num_integrated),position_y_integrated(1:num_integrated),'-or','LineWidth',2);
    axis equal 
  end
  title('Visual odometry results by LIBVISO2(blue) and BioVO(red)');
  xlabel('x: m');
  ylabel('y: m');
%   pause(0.05); refresh;

  % output statistics
  num_matches = visualOdometryStereoMex('num_matches');
  num_inliers = visualOdometryStereoMex('num_inliers');
end

% release visual odometry
visualOdometryStereoMex('close');
