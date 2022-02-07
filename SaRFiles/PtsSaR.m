function loc = PtsSaR(drc8, xc, yc)
% Input
% 	drc8  : 8*1 vector, the length of 8 directions LPA-ICI found
%   xc    : the x coordinate of the center pixel
%   yc    : the y coordinate of the center pixel
% Output
%   loc   : x, y coordinates of pixels in (xc,yc)'s shape-adaptive region

drc8 = double(drc8);
xc = double(xc);
yc = double(yc);

step = drc8-1;

% Points in searching region
wSize = max(step);
pts_x = repmat([-wSize:wSize],2*wSize+1,1);
pts_x = pts_x(:);
pts_y = repmat([-wSize:wSize]',1,2*wSize+1);
pts_y = pts_y(:);

% Vertices of polygon
x = zeros(9,1);
y = zeros(9,1);
x(1) = 0;        y(1) = -step(1);
x(2) = step(2);  y(2) = -step(2);
x(3) = step(3);  y(3) = 0;
x(4) = step(4);  y(4) = step(4);
x(5) = 0;        y(5) = step(5);
x(6) = -step(6); y(6) = step(6);
x(7) = -step(7); y(7) = 0;
x(8) = -step(8); y(8) = -step(8);
x(9) = 0;        y(9) = -step(1);

% Find all points inside the polygon
selected = zeros((2*wSize+1).^2,8);
for i=1:8
    if polyarea([x(i:i+1);0],[y(i:i+1);0])>0
        selected(:,i) = inpolygon(pts_x,pts_y,[x(i:i+1);0],[y(i:i+1);0]);
    end
end

xs = [pts_x(sum(selected,2)>0);x]+xc;
ys = [pts_y(sum(selected,2)>0);y]+yc;

loc = unique([xs,ys], 'rows');

end