close all;

%Scatter plot  Fig1 (original)
font_size = 10;

%figure('Color',[1 1 1]);

figure;
set(gca,'FontSize',font_size);

scatter(scatter_Head_speed, scatter_Frame_size,10,'MarkerFaceColor',[0 0.450980392156863 0.741176470588235]);
hold on;

% Linear regression
[r,m,b] = regression(scatter_Head_speed', scatter_Frame_size');
x = 0:1:60;
plot(x, b + m * x,'LineWidth',1,'LineStyle', '-','Color','r');

% Quadratic regression
mdl = fitlm(scatter_Head_speed,scatter_Frame_size,'purequadratic','RobustOpts','on');
a0 = mdl.Coefficients.Estimate(1);
a1 = mdl.Coefficients.Estimate(2);
a2 = mdl.Coefficients.Estimate(3);
plot(x, a0 + a1 * x + a2 * x.^2,'LineWidth',1,'LineStyle', '-','Color','b');

ylabel({'Frame size (kbytes)'});
xlabel({'Head speed (deg/s)'});
axis([0 60 800 2200]);
legend('Data sample','Linear regression','Quadratic regression','Location','southeast','FontSize',font_size);
box on;
grid on;
set(gca,'fontname','times');


