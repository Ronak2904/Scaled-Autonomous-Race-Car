clc
A = [iopt(1:4);
    iopt(5:8);
    iopt(9:12);
    iopt(13:16)];
B = [iopt(17);
iopt(18);
iopt(19);
iopt(20)];

C = [0 0 0 1];
D = 0;

sys = ss(A,B,C,D);
eig(A)
x_exp = readmatrix('sine_1_7_22_22.xlsx');
t = x_exp(:,1);
u = x_exp(:,11);
x0 = [0 0 0 0];
x_sim = lsim(sys,u,t,x0);
x_cal = x_exp(:,14);
plot(t,x_sim,t,x_cal)
figure (1)
plot(t,x_sim(:,1),t,x_cal(:,1))
figure (2)
plot(t,x_sim(:,2),t,x_cal(:,2))
figure (3)
plot(t,x_sim(:,3),t,x_cal(:,3))