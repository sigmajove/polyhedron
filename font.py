class Font:
    # SPACING is the distance between the two glyphs of
    # a two-digit number.
    SPACING = 100

    def __init__(self, pen):
        self.pen = pen

    # All the Draw routines make calls to self.pen to write out
    # the digits, and return a tuple (x, y) of the dimensions
    # of what was drawn.

    def d0(self):
        self.pen.move_to((706.00, 747.00))
        self.pen.curve_to((666.50, 1135.00), (706.00, 1009.00))
        self.pen.curve_to((500.00, 1261.00), (627.00, 1261.00))
        self.pen.curve_to((332.00, 1135.00), (373.00, 1261.00))
        self.pen.curve_to((291.00, 747.00), (291.00, 1009.00))
        self.pen.curve_to((332.00, 364.50), (291.00, 497.00))
        self.pen.curve_to((500.00, 232.00), (373.00, 232.00))
        self.pen.curve_to((666.50, 364.50), (627.00, 232.00))
        self.pen.curve_to((706.00, 747.00), (706.00, 497.00))
        self.pen.close_path(hole=True)

        self.pen.move_to((1000.00, 747.00))
        self.pen.curve_to((888.50, 194.00), (1000.00, 388.00))
        self.pen.curve_to((500.00, 0.00), (777.00, 0.00))
        self.pen.curve_to((111.50, 194.00), (223.00, 0.00))
        self.pen.curve_to((0.00, 747.00), (0.00, 388.00))
        self.pen.curve_to((111.50, 1301.00), (0.00, 1106.00))
        self.pen.curve_to((500.00, 1496.00), (223.00, 1496.00))
        self.pen.curve_to((888.50, 1301.00), (777.00, 1496.00))
        self.pen.curve_to((1000.00, 747.00), (1000.00, 1106.00))
        self.pen.close_path()
        return (1000.00, 1496.00)

    def add_extras(self, p0, p1, n):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        for i in range (1, n):
            self.pen.add_steiner((p0[0] + (i * dx) / n, p0[1] + (i * dy) / n))

    def d1(self):
        self.pen.move_to((0.00, 1036.00))
        self.pen.line_to((0.00, 1230.00))
        self.pen.curve_to((189.00, 1248.00), (135.00, 1236.00))
        self.pen.curve_to((329.00, 1324.00), (275.00, 1267.00))
        self.pen.curve_to((385.00, 1428.00), (366.00, 1363.00))
        self.pen.curve_to((396.00, 1496.00), (396.00, 1467.00))
        # self.add_extras((0.00, 1230.00), (396.0, 1496.0), 5)
        # self.add_extras((0.00, 1230.00+300), (396.0, 1496.0+300), 4)
        self.pen.line_to((633.00, 1492.00))
        self.pen.line_to((633.00, 0.00))
        self.pen.line_to((341.00, 0.00))
        self.pen.line_to((341.00, 1036.00))
        self.pen.close_path()
        return (633.00, 1496.00)

    def d2(self):
        self.pen.move_to((67.00, 321.00))
        self.pen.curve_to((355.00, 628.00), (128.00, 466.00))
        self.pen.curve_to((610.00, 830.00), (552.00, 769.00))
        self.pen.curve_to((699.00, 1038.00), (699.00, 925.00))
        self.pen.curve_to((648.00, 1191.00), (699.00, 1130.00))
        self.pen.curve_to((502.00, 1252.00), (597.00, 1252.00))
        self.pen.curve_to((325.00, 1155.00), (372.00, 1252.00))
        self.pen.curve_to((293.00, 977.00), (298.00, 1099.00))
        self.pen.line_to((16.00, 977.00))
        self.pen.curve_to((83.00, 1276.00), (23.00, 1162.00))
        self.pen.curve_to((488.00, 1496.00), (197.00, 1496.00))
        self.pen.curve_to((854.00, 1365.50), (718.00, 1496.00))
        self.pen.curve_to((990.00, 1028.00), (990.00, 1238.00))
        self.pen.curve_to((894.00, 742.00), (990.00, 867.00))
        self.pen.curve_to((687.00, 557.00), (831.00, 659.00))
        self.pen.line_to((573.00, 476.00))
        self.pen.curve_to((426.50, 366.00), (466.00, 400.00))
        self.pen.curve_to((360.00, 251.00), (387.00, 332.00))
        self.pen.line_to((993.00, 251.00))
        self.pen.line_to((993.00, 0.00))
        self.pen.line_to((0.00, 0.00))
        self.pen.curve_to((67.00, 321.00), (4.00, 192.00))
        self.pen.close_path()
        return (993.00, 1496.00)

    def d3(self):
        self.pen.move_to((280.00, 481.00))
        self.pen.curve_to((308.00, 337.00), (280.00, 394.00))
        self.pen.curve_to((497.00, 232.00), (360.00, 232.00))
        self.pen.curve_to((643.50, 289.50), (581.00, 232.00))
        self.pen.curve_to((706.00, 455.00), (706.00, 347.00))
        self.pen.curve_to((590.00, 646.00), (706.00, 598.00))
        self.pen.curve_to((382.00, 673.00), (524.00, 673.00))
        self.pen.line_to((382.00, 877.00))
        self.pen.curve_to((576.00, 904.00), (521.00, 879.00))
        self.pen.curve_to((671.00, 1074.00), (671.00, 946.00))
        self.pen.curve_to((622.50, 1209.00), (671.00, 1157.00))
        self.pen.curve_to((486.00, 1261.00), (574.00, 1261.00))
        self.pen.curve_to((337.50, 1197.00), (385.00, 1261.00))
        self.pen.curve_to((292.00, 1026.00), (290.00, 1133.00))
        self.pen.line_to((26.00, 1026.00))
        self.pen.curve_to((63.00, 1231.00), (30.00, 1134.00))
        self.pen.curve_to((173.00, 1388.00), (98.00, 1316.00))
        self.pen.curve_to((306.00, 1466.00), (229.00, 1439.00))
        self.pen.curve_to((495.00, 1496.00), (383.00, 1496.00))
        self.pen.curve_to((830.50, 1385.50), (703.00, 1496.00))
        self.pen.curve_to((958.00, 1097.00), (958.00, 1278.00))
        self.pen.curve_to((882.00, 881.00), (958.00, 969.00))
        self.pen.curve_to((782.00, 806.00), (834.00, 826.00))
        self.pen.curve_to((894.00, 739.00), (821.00, 806.00))
        self.pen.curve_to((1003.00, 463.00), (1003.00, 638.00))
        self.pen.curve_to((875.50, 139.50), (1003.00, 279.00))
        self.pen.curve_to((498.00, 0.00), (748.00, 0.00))
        self.pen.curve_to((70.00, 201.00), (190.00, 0.00))
        self.pen.curve_to((0.00, 481.00), (7.00, 308.00))
        self.pen.close_path()
        return (1003.00, 1496.00)

    def d4(self):
        y0 = 0
        y1 = 348
        y2 = 605
        y3 = 1200
        y4 = 1496
        x0 = 0
        x1 = 270
        x2 = 533
        x3 = 632
        x4 = 890
        x5 = 1120
        self.pen.move_to((x3, y1))

        self.pen.line_to((x0, y1))
        self.pen.line_to((x0, y2))
        self.pen.line_to((x2, y4))
        self.pen.line_to((x4, y4))
        self.pen.line_to((x4, y2))
        self.pen.line_to((x5, y2))
        self.pen.line_to((x5, y1))
        self.pen.line_to((x4, y1))
        self.pen.line_to((x4, y0))
        self.pen.line_to((x3, y0))
        self.pen.close_path()

        # self.pen.move_to((x3, y1))
        self.pen.move_to((x3, y2))
        self.pen.line_to((x3, y3))
        self.pen.line_to((x1, y2))
        self.pen.line_to((x3, y2))
        self.pen.close_path(hole=True)
        return (1120.00, 1496.00)

    def d5(self):
        self.pen.move_to((284.00, 424.00))
        self.pen.curve_to((349.00, 280.50), (301.00, 331.00))
        self.pen.curve_to((489.00, 230.00), (397.00, 230.00))
        self.pen.curve_to((650.50, 304.50), (595.00, 230.00))
        self.pen.curve_to((706.00, 492.00), (706.00, 379.00))
        self.pen.curve_to((654.00, 679.50), (706.00, 603.00))
        self.pen.curve_to((492.00, 756.00), (602.00, 756.00))
        self.pen.curve_to((402.00, 743.00), (440.00, 756.00))
        self.pen.curve_to((301.00, 654.00), (335.00, 719.00))
        self.pen.line_to((45.00, 666.00))
        self.pen.line_to((147.00, 1496.00))
        self.pen.line_to((946.00, 1496.00))
        self.pen.line_to((946.00, 1225.00))
        self.pen.line_to((353.00, 1225.00))
        self.pen.line_to((301.00, 908.00))
        self.pen.curve_to((404.00, 965.00), (367.00, 951.00))
        self.pen.curve_to((555.00, 988.00), (466.00, 988.00))
        self.pen.curve_to((869.00, 867.00), (735.00, 988.00))
        self.pen.curve_to((1003.00, 515.00), (1003.00, 746.00))
        self.pen.curve_to((874.00, 156.00), (1003.00, 314.00))
        self.pen.curve_to((488.00, 0.00), (745.00, 0.00))
        self.pen.curve_to((148.00, 109.00), (281.00, 0.00))
        self.pen.curve_to((0.00, 424.00), (15.00, 220.00))
        self.pen.close_path()
        return (1003.00, 1496.00)

    def d6(self):
        self.pen.move_to((286, 495))
        self.pen.curve_to((349.00, 304.00), (286.00, 378.00))
        self.pen.curve_to((509.00, 230.00), (412.00, 230.00))
        self.pen.curve_to((658.50, 301.50), (604.00, 230.00))
        self.pen.curve_to((713.00, 487.00), (713.00, 373.00))
        self.pen.curve_to((651.00, 681.50), (713.00, 614.00))
        self.pen.curve_to((499.00, 749.00), (589.00, 749.00))
        self.pen.curve_to((370.00, 705.00), (426.00, 749.00))
        self.pen.curve_to((286.00, 495.00), (286.00, 640.00))
        self.pen.close_path(hole=True)

        self.pen.move_to((0, 689))
        self.pen.curve_to((14.00, 959.00), (0.00, 855.00))
        self.pen.curve_to((111.00, 1267.00), (39.00, 1144.00))
        self.pen.curve_to((273.50, 1436.00), (173.00, 1372.00))
        self.pen.curve_to((514.00, 1496.00), (374.00, 1496.00))
        self.pen.curve_to((836.00, 1396.50), (716.00, 1496.00))
        self.pen.curve_to((971.00, 1121.00), (956.00, 1293.00))
        self.pen.line_to((700, 1158))
        self.pen.curve_to((521.00, 1266.00), (654.00, 1266.00))
        self.pen.curve_to((323.00, 1110.00), (382.00, 1266.00))
        self.pen.curve_to((279.00, 856.00), (291.00, 1024.00))
        self.pen.curve_to((402.00, 948.00), (332.00, 919.00))
        self.pen.curve_to((562.00, 977.00), (472.00, 977.00))
        self.pen.curve_to((878.50, 846.00), (755.00, 977.00))
        self.pen.curve_to((1002.00, 511.00), (1002.00, 715.00))
        self.pen.curve_to((881.00, 153.00), (1002.00, 308.00))
        self.pen.curve_to((505.00, 0.00), (760.00, 0.00))
        self.pen.curve_to((101.00, 227.00), (231.00, 0.00))
        self.pen.curve_to((0.00, 689.00), (0.00, 406.00))
        self.pen.close_path()
        return (1002.00, 1496.00)

    def d7(self):
        self.pen.move_to((1028.00, 1244.00))
        self.pen.curve_to((850.00, 1019.50), (964.00, 1181.00))
        self.pen.curve_to((659.00, 686.00), (736.00, 858.00))
        self.pen.curve_to((549.00, 356.00), (598.00, 551.00))
        self.pen.curve_to((500.00, 0.00), (500.00, 161.00))
        self.pen.line_to((204.00, 0.00))
        self.pen.curve_to((460.00, 847.00), (217.00, 426.00))
        self.pen.curve_to((680.00, 1246.00), (617.00, 1108.00))
        self.pen.line_to((0.00, 1246.00))
        self.pen.line_to((4.00, 1496.00))
        self.pen.line_to((1028.00, 1496.00))
        self.pen.close_path()
        return (1028.00, 1496.00)

    def d8(self):
        self.pen.move_to((218.00, 808.00))
        self.pen.curve_to((81.50, 959.50), (113.00, 878.00))
        self.pen.curve_to((50.00, 1112.00), (50.00, 1041.00))

        self.pen.curve_to((169.00, 1381.50), (50.00, 1270.00))
        self.pen.curve_to((505.00, 1496.00), (288.00, 1496.00))
        self.pen.curve_to((841.00, 1381.50), (722.00, 1496.00))
        self.pen.curve_to((960.00, 1112.00), (960.00, 1270.00))
        self.pen.curve_to((928.50, 959.50), (960.00, 1041.00))
        self.pen.curve_to((792.00, 818.00), (897.00, 878.00))
        self.pen.curve_to((953.00, 659.00), (899.00, 758.00))
        self.pen.curve_to((1007.00, 438.00), (1007.00, 560.00))
        self.pen.curve_to((871.50, 126.50), (1007.00, 255.00))
        self.pen.curve_to((493.00, 0.00), (736.00, 0.00))
        self.pen.curve_to((125.00, 126.50), (250.00, 0.00))
        self.pen.curve_to((0.00, 438.00), (0.00, 255.00))
        self.pen.curve_to((55.50, 659.00), (0.00, 560.00))
        self.pen.curve_to((218.00, 808.00), (111.00, 758.00))
        self.pen.close_path()

        self.pen.move_to((322, 1080))
        self.pen.curve_to((370.50, 951.00), (322.00, 1001.00))
        self.pen.curve_to((505.00, 901.00), (419.00, 901.00))
        self.pen.curve_to((639.50, 951.00), (592.00, 901.00))
        self.pen.curve_to((687.00, 1080.00), (687.00, 1001.00))
        self.pen.curve_to((639.50, 1214.50), (687.00, 1166.00))
        self.pen.curve_to((505.00, 1263.00), (592.00, 1263.00))
        self.pen.curve_to((370.50, 1214.50), (419.00, 1263.00))
        self.pen.curve_to((322.00, 1080.00), (322.00, 1166.00))
        self.pen.close_path(hole=True)

        self.pen.move_to((296, 457))
        self.pen.curve_to((351.50, 291.00), (296.00, 350.00))
        self.pen.curve_to((505.00, 232.00), (407.00, 232.00))
        self.pen.curve_to((658.50, 291.00), (603.00, 232.00))
        self.pen.curve_to((714.00, 457.00), (714.00, 350.00))
        self.pen.curve_to((657.50, 625.50), (714.00, 568.00))
        self.pen.curve_to((505.00, 683.00), (601.00, 683.00))
        self.pen.curve_to((352.50, 625.50), (409.00, 683.00))
        self.pen.curve_to((296.00, 457.00), (296.00, 568.00))
        self.pen.close_path(hole=True)

        return (1007.00, 1496.00)

    def d9(self):
        self.pen.move_to((484.00, 1496.00))
        self.pen.curve_to((938.00, 1205.00), (815.00, 1496.00))
        self.pen.curve_to((1008.00, 768.00), (1008.00, 1039.00))
        self.pen.curve_to((941.00, 329.00), (1008.00, 505.00))
        self.pen.curve_to((471.00, 0.00), (813.00, 0.00))
        self.pen.curve_to((178.00, 90.50), (308.00, 0.00))
        self.pen.curve_to((29.00, 372.00), (48.00, 187.00))
        self.pen.line_to((313.00, 372.00))
        self.pen.curve_to((367.00, 268.00), (323.00, 308.00))
        self.pen.curve_to((484.00, 228.00), (411.00, 228.00))
        self.pen.curve_to((682.00, 384.00), (625.00, 228.00))
        self.pen.curve_to((721.00, 635.00), (713.00, 470.00))
        self.pen.curve_to((638.00, 560.00), (682.00, 586.00))
        self.pen.curve_to((441.00, 512.00), (558.00, 512.00))
        self.pen.curve_to((134.00, 631.50), (268.00, 512.00))
        self.pen.curve_to((0.00, 976.00), (0.00, 751.00))
        self.pen.line_to((289.00, 1002.00))

        self.pen.curve_to((341.50, 808.50), (289.00, 873.00))
        self.pen.curve_to((503.00, 744.00), (394.00, 744.00))
        self.pen.curve_to((614.00, 778.00), (562.00, 744.00))
        self.pen.curve_to((711.00, 993.00), (711.00, 840.00))
        self.pen.curve_to((653.50, 1188.00), (711.00, 1116.00))
        self.pen.curve_to((496.00, 1260.00), (596.00, 1260.00))
        self.pen.curve_to((371.00, 1219.00), (423.00, 1260.00))
        self.pen.curve_to((289.00, 1002.00), (289.00, 1155.00))

        self.pen.line_to((0.00, 976.00))
        self.pen.curve_to((134.50, 1353.50), (0.00, 1209.00))
        self.pen.curve_to((484.00, 1496.00), (269.00, 1496.00))

        self.pen.close_path()
        return (1008.00, 1496.00)

    def dot(self):
        radius = 150.0
        self.pen.move_to((radius, 0.0))
        self.pen.curve_to((2 * radius, radius), (2 * radius, 0))
        self.pen.curve_to((radius, 2 * radius), (2 * radius, 2 * radius))
        self.pen.curve_to((0, radius), (0, 2 * radius))
        self.pen.curve_to((radius, 0), (0, 0))
        self.pen.close_path()
        return (2 * radius, 2 * radius)

    def draw(self, digit):
        match digit:
            case 0:
                return self.d0()
            case 1:
                return self.d1()
            case 2:
                return self.d2()
            case 3:
                return self.d3()
            case 4:
                return self.d4()
            case 5:
                return self.d5()
            case 6:
                x1, y1 = self.d6()
                x1 += 40
                self.pen.advance(x1)
                x2, y2 = self.dot()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 7:
                return self.d7()
            case 8:
                return self.d8()
            case 9:
                x1, y1 = self.d9()
                x1 += 40
                self.pen.advance(x1)
                x2, y2 = self.dot()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 10:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d0()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 11:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d1()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 12:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d2()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 13:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d3()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 14:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d4()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 15:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d5()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 16:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d6()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 17:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d7()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 18:
                x1, y1 = self.d1()
                x1 += Font.SPACING
                self.pen.advance(x1)
                x2, y2 = self.d8()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))

            case _:
                raise RuntimeError(f"(Bad digit {digit}")

