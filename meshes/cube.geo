cl = 1./100.;

Point(1) = {0, 0, 0, cl};
Point(2) = {1, 0, 0, cl};
Point(3) = {1, 1, 0, cl};
Point(4) = {0, 1, 0, cl};
Point(5) = {0, 0, 1, cl};
Point(6) = {1, 0, 1, cl};
Point(7) = {1, 1, 1, cl};
Point(8) = {0, 1, 1, cl};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {5, 1};
Line(10) = {6, 2};
Line(11) = {7, 3};
Line(12) = {8, 4};
Point(9) = {0.4, 0.4, 0.4, cl};
Point(10) = {0.6, 0.4, 0.4, cl};
Point(11) = {0.6, 0.6, 0.4, cl};
Point(12) = {0.4, 0.6, 0.4, cl};
Point(13) = {0.4, 0.4, 0.6, cl};
Point(14) = {0.6, 0.4, 0.6, cl};
Point(15) = {0.6, 0.6, 0.6, cl};
Point(16) = {0.4, 0.6, 0.6, cl};
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};
Line(17) = {13, 14};
Line(18) = {14, 15};
Line(19) = {15, 16};
Line(20) = {16, 13};
Line(21) = {13, 9};
Line(22) = {14, 10};
Line(23) = {16, 12};
Line(24) = {15, 11};

// INTERNAL
Line Loop(25) = {18, 24, -14, -22};
Plane Surface(26) = {25};
Line Loop(27) = {-13, 22, 17, -21};
Plane Surface(28) = {27};
Line Loop(29) = {-20, -17, -18, -19};
Plane Surface(30) = {29};
Line Loop(31) = {13, 14, 15, 16};
Plane Surface(32) = {31};
Line Loop(33) = {19, 23, -15, -24};
Plane Surface(34) = {33};
Line Loop(35) = {-23, -16, 21, 20};
Plane Surface(36) = {35};

Surface Loop(37) = {36, 34, 30, 28, 32, 26};

// COMO O PROBLEMA EM QUESTAO TRATA-SE DE UM CUBO COM UMA PARTE OCA INTERNA, ENTÃO NÃO SERÁ
// CRIADO NENHUM VOLUME, POIS ISTO LEVARIA A CRIAÇÃO DE ELEMENTO NESTE BURACO O QUE É
// ALGO QUE NÃO DESEJAMOS.

// Volume(38) = {37}; ESTA LINHA DEVE ESTAR COMENTADA!!!!


// EXTERNAL
Line Loop(39) = {3, -12, -7, 11};
Plane Surface(40) = {39};
Line Loop(41) = {-8, -9, 4, 12};
Plane Surface(42) = {41};
Line Loop(43) = {-4, -1, -2, -3};
Plane Surface(44) = {43};
Line Loop(45) = {1, -10, -5, 9};
Plane Surface(46) = {45};
Line Loop(47) = {-6, -11, 2, 10};
Plane Surface(48) = {47};
Line Loop(49) = {7, 8, 5, 6};
Plane Surface(50) = {49};

// surface loop
Surface Loop(51) = {40, 44, 42, 50, 46, 48};
Volume(1) = {51,-37};

Physical Surface(10) = {40, 44, 42, 50, 46, 48};	// EXTERNA
Physical Surface(51) = {36, 34, 30, 28, 32, 26};	// INTERNA
Physical Volume(1) = {1};	// MEIO HOMOGENEO



Transfinite Line {9, 5, 10, 1, 8, 6, 7, 4, 2, 3, 12, 11} = 5 Using Progression 1;
Transfinite Line {21, 17, 22, 13, 20, 18, 19, 24, 23, 15, 16, 14} = 2 Using Progression 1;

Transfinite Surface {26};
Transfinite Surface {28};
Transfinite Surface {30};
Transfinite Surface {32};
Transfinite Surface {34};
Transfinite Surface {36};
Transfinite Surface {40};
Transfinite Surface {42};
Transfinite Surface {44};
Transfinite Surface {46};
Transfinite Surface {48};
Transfinite Surface {50};
