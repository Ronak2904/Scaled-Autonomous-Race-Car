%Calling fmincon%

%Initialization
x0 = -0.1*rand(1,20);

% Setting the parameters for the objective function
options = optimset('PlotFcns',@optimplotfval);

% Initializing the minimization search function using the parameters defined and
% the initial values for the target variables.
iopt = fminsearch(@objective,x0,options);

%Obejctive Function%
function obj = objective(i)
    
a11 = i(1);

a12 = i(2);

a13 = i(3);

a14 = i(4);

a21 = i(5);

a22 = i(6);

a23 = i(7);

a24 = i(8);

a31 = i(9);

a32 = i(10);

a33 = i(11);

a34 = i(12);

a41 = i(13);

a42 = i(14);

a43 = i(15);

a44 = i(16);

b11 = i(17);

b21 = i(18);

b31 = i(19);

b41 = i(20);

x_exp = readmatrix('sine_1_7_22_22.xlsx');

t = x_exp(:,1);

u = x_exp(:,11);

x_cal = x_exp(:,14);

x0 = [0 0 0 0];

A = [a11 a12 a13 a14;
    a21 a22 a23 a24;
    a31 a32 a33 a34;
    a41 a42 a43 a44];

B = [b11 ;
    b21  ;
    b31  ;
    b41  ];

C = [ 0 0 0 1];

D = 0;

sys = ss(A,B,C,D);

x_sim = lsim(sys,u,t,x0);

Error = x_cal - x_sim;

obj = trace(Error' * Error);

end 

