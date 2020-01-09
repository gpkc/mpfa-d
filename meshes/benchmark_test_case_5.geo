// Gmsh project created on Mon Jan 28 16:03:43 2019
cl = 0.5;
Point(1) = {0, 0, 0, cl};
Point(2) = {1, 0, 0, cl};
Point(3) = {0, 1, 0, cl};
Point(4) = {1, 1, 0, cl};
Point(5) = {0, 0, 1, cl};
Point(6) = {1, 0, 1, cl};
Point(7) = {0, 1, 1, cl};
Point(8) = {1, 1, 1, cl};
Point(9) = {0.5, 0.5, 0, cl};
Point(11) = {0.5, 0.5, 1, cl};
Point(12) = {0.5, 0, 0.5, cl};
Point(13) = {0.5, 1, 0.5, cl};
Point(14) = {0, 0.5, 0.5, cl};
Point(15) = {1, 0.5, 0.5, cl};
Point(10) = {0.5, 0.5, 0.5, cl};
Point(16) = {1, 0, 0.5, cl};
Point(17) = {1, 1, 0.5, cl};
Point(18) = {0, 1, 0.5, cl};
Point(19) = {0, 0, 0.5, cl};
Point(20) = {0, 0.5, 0, cl};
Point(21) = {0, 0.5, 1, cl};
Point(22) = {1, 0.5, 1, cl};
Point(23) = {0, 0.5, 1, cl};
Point(24) = {1, 0.5, 0, cl};
Point(25) = {0.5, 0, 0, cl};
Point(26) = {0.5, 0, 1, cl};
Point(27) = {0.5, 1, 1, cl};
Point(28) = {0.5, 1, 0, cl};
Line(13) = {21, 14};
Line(14) = {19, 14};
Line(15) = {14, 20};
Line(16) = {14, 18};
Line(17) = {21, 11};
Line(18) = {11, 22};
Line(19) = {11, 26};
Line(20) = {11, 27};
Line(21) = {22, 15};
Line(22) = {15, 24};
Line(23) = {16, 15};
Line(24) = {17, 15};
Line(25) = {27, 13};
Line(26) = {13, 28};
Line(27) = {24, 9};
Line(28) = {20, 9};
Line(29) = {25, 9};
Line(30) = {28, 9};
Line(31) = {25, 12};
Line(32) = {12, 26};
Line(33) = {19, 12};
Line(34) = {16, 12};
Line(35) = {11, 10};
Line(36) = {9, 10};
Line(37) = {14, 10};
Line(38) = {10, 15};
Line(39) = {13, 10};
Line(40) = {18, 13};
Line(41) = {13, 17};
Line(42) = {12, 10};
Line(67) = {8, 17};
Line(68) = {17, 4};
Line(69) = {4, 28};
Line(70) = {28, 3};
Line(71) = {3, 18};
Line(72) = {18, 7};
Line(73) = {7, 27};
Line(74) = {27, 8};
Line(75) = {8, 22};
Line(76) = {22, 6};
Line(77) = {6, 16};
Line(78) = {16, 2};
Line(79) = {2, 25};
Line(80) = {25, 1};
Line(81) = {1, 20};
Line(82) = {20, 3};
Line(83) = {24, 4};
Line(84) = {2, 24};
Line(85) = {1, 19};
Line(86) = {19, 5};
Line(87) = {5, 26};
Line(88) = {26, 6};
Line(89) = {5, 21};
Line(90) = {21, 7};
Delete {
  Line{36, 35, 26, 25, 31, 32, 42, 39};
}
Delete {
  Line{38, 37, 41, 40, 34, 33};
}
Delete {
  Line{74, 73, 18, 17, 88, 87, 69, 70, 27, 28, 79, 80};
}
Delete {
  Line{30, 29, 20, 19};
}
Delete {
  Point{28, 13, 27, 11, 26, 12, 25, 9, 10};
}
Line(91) = {3, 4};
Line(92) = {18, 17};
Line(93) = {7, 8};
Line(94) = {21, 22};
Line(95) = {5, 6};
Line(96) = {19, 16};
Line(97) = {1, 2};
Line(98) = {20, 24};
Line(99) = {14, 15};
Line Loop(100) = {15, 98, -22, -99};
Plane Surface(101) = {100};
Line Loop(102) = {99, -21, -94, 13};
Plane Surface(103) = {102};
Line Loop(104) = {99, -24, -92, -16};
Plane Surface(105) = {104};
Line Loop(106) = {23, -99, -14, 96};
Plane Surface(107) = {106};
Line Loop(108) = {77, -96, 86, 95};
Plane Surface(109) = {108};
Line Loop(110) = {96, 78, -97, 85};
Plane Surface(111) = {110};
Line Loop(112) = {67, -92, 72, 93};
Plane Surface(113) = {112};
Line Loop(114) = {92, 68, -91, 71};
Plane Surface(115) = {114};
Line Loop(116) = {24, 22, 83, -68};
Plane Surface(117) = {116};
Line Loop(118) = {84, -22, -23, 78};
Plane Surface(119) = {118};
Line Loop(120) = {23, -21, 76, 77};
Line Loop(121) = {24, -21, -75, 67};
Plane Surface(122) = {121};
Plane Surface(123) = {120};
Line Loop(124) = {16, 72, -90, 13};
Plane Surface(125) = {124};
Line Loop(126) = {13, -14, 86, 89};
Plane Surface(127) = {126};
Line Loop(128) = {15, -81, 85, 14};
Plane Surface(129) = {128};
Line Loop(130) = {15, 82, 71, -16};
Plane Surface(131) = {130};
Line Loop(132) = {90, 93, 75, -94};
Plane Surface(133) = {132};
Line Loop(134) = {76, -95, 89, 94};
Plane Surface(135) = {134};
Line Loop(136) = {91, -83, -98, 82};
Plane Surface(137) = {136};
Line Loop(138) = {98, -84, -97, 81};
Plane Surface(139) = {138};
Surface Loop(146) = {135, 123, 109, 127, 107, 103};
Volume(4) = {146};
Surface Loop(144) = {113, 122, 133, 125, 105, 103};
Volume(3) = {144};
Surface Loop(142) = {101, 137, 115, 117, 131, 105};
Volume(2) = {142};
Surface Loop(140) = {129, 139, 119, 111, 107, 101};
Volume(1) = {140};
//+
Physical Surface(101) = {123, 135, 133, 113, 115, 137, 139, 111, 109, 119, 122, 117, 131, 125, 127, 129};
//+
Physical Volume(1) = {1};
//+
Physical Volume(2) = {2};
//+
Physical Volume(3) = {3};
//+
Physical Volume(4) = {4};
