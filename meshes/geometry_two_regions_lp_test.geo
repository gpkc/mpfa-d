// Gmsh project created on Wed Feb 21 10:01:44 2018
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0.5, 0, 0, 1.0}; //
Point(3) = {1, 0, 0, 1.0};
Point(4) = {1, 1, 0, 1.0};
Point(5) = {0.5, 1, 0, 1.0}; //
Point(6) = {0, 1, 0, 1.0};
Point(7) = {0, 0, 1, 1.0};
Point(8) = {0.5, 0, 1, 1.0}; //
Point(9) = {1, 0, 1, 1.0};
Point(10) = {1, 1, 1, 1.0};
Point(11) = {0.5, 1, 1, 1.0}; //
Point(12) = {0, 1, 1, 1.0};

Line(1) = {1, 6};
Line(2) = {6, 12};
Line(3) = {12, 7};
Line(4) = {7, 1};
Line(5) = {2, 5};
Line(6) = {5, 11};
Line(7) = {11, 8};
Line(8) = {8, 2};
Line(9) = {3, 4};
Line(10) = {4, 10};
Line(11) = {10, 9};
Line(12) = {9, 3};
Line(13) = {1, 2};
Line(14) = {8, 7};
Line(15) = {6, 5};
Line(16) = {11, 12};
Line(17) = {2, 3};
Line(18) = {9, 8};
Line(19) = {5, 4};
Line(20) = {10, 11};

Line Loop(21) = {4, 1, 2, 3};
Plane Surface(22) = {21};

Line Loop(23) = {4, 13, -8, 14};
Plane Surface(24) = {23};

Line Loop(25) = {1, 15, -5, -13};
Plane Surface(26) = {25};

Line Loop(27) = {2, -16, -6, -15};
Plane Surface(28) = {27};

Line Loop(29) = {3, -14, -7, 16};
Plane Surface(30) = {29};

Line Loop(31) = {8, 17, -12, 18};
Plane Surface(32) = {31};

Line Loop(33) = {17, 9, -19, -5};
Plane Surface(34) = {33};

Line Loop(35) = {8, 5, 6, 7};
Plane Surface(36) = {35};

Line Loop(37) = {12, 9, 10, 11};
Plane Surface(38) = {37};

Line Loop(39) = {19, 10, 20, -6};
Plane Surface(40) = {39};

Line Loop(41) = {18, -7, -20, 11};
Plane Surface(42) = {41};

Surface Loop(43) = {24, 22, 26, 28, 30, 36};
Volume(44) = {43};

Surface Loop(45) = {38, 32, 34, 40, 42, 36};
Volume(46) = {45};

Physical Surface(101) = {22, 38, 24, 28, 32, 40, 26, 30, 34, 42 };
//Physical Surface(101) = {26, 30, 34, 42};

Physical Volume(1) = {44};
Physical Volume(2) = {46};
