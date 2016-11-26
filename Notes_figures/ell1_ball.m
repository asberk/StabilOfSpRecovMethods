%% Making a graphic
clear all; close all; clc;
Vx = [0, 1, 0, -1];
Vy = [1, 0, -1, 0];
fill(Vx, Vy, [.9, .95, 1]);

hold on;

plot(0, .75, '.k');
text(0, .67, '$x$', 'Interpreter', 'latex', 'FontSize', 14);

plot(.2, 1.4, '.k');
text(.2, 1.3, '$x+z_1$', 'Interpreter', 'latex', 'FontSize', 14);
plot(.2, 1, '.k');
text(.2, .9, '$x+z_2$', 'Interpreter', 'latex', 'FontSize', 14);

L = linspace(-1, 1, 3);
plot(L, abs(L)+.75, 'r-.');
plot(L, abs(L)+1, 'r-.');

xlim([-1.5, 1.5]);
ylim([-1.5, 1.5]);
axis('square');
