标准单元工艺统一规定：
1. 线网的terminal都分布在标准单元的上边界和下边界，每个晶体管terminal按照source，gate，drain 排列，然后相邻的晶体管必定是前一个的drain 等于后一个的 source，这样假设一排有n个晶体管，那么M0的网格就会有2*n-1列，从0开始，偶数列是S/D列，基数列是Gate列。
   
2. source列和drain列只能竖着布线，不能横向布线。并且，上半区的source/drain和下半区的source/drain在M0是不可以通过中间联通的（因为有Active区域的存在，规定了source/drain在M0的竖向布线区域），而gate的话也只能在Active区域外能有横向走线。如果上下两排有相同的gate，那么他们是可以直接通过M0进行连通的，特别是上下对应的晶体管的gate相同，我们叫它共栅，是可以直接相连的；但是若距离比较远，那更好的选择是通过M1或者M2的走线，从上层进行连接，M0尽量保持精简。
   
3. gate共栅的话则必须相连

4. 没有填S/D/G的位置则是None，表示的是当前的晶体管可能是dummy或者break，这里给出dummy、break的判断标准：
   1. 若gate == None，则是dummy或者break
   2. 在1.的情况下，若该晶体管的S==D（或者S和D一个是None一个不是），则是dummy，否则是break

5. Break的晶体管的G不可以有任何布线，Dummy的晶体管的G根据工艺的定义可能会有不同规则，但默认不布线



YMTC工艺特定：

1. 读取techs/ymtc_cj3/grid_place中的 [
    AND2_X1_ML.xlsx,
]
这个（些）文件描述了YMTC的CJ3工艺的标准单元在M0层的网格布局情况（S/D Active）。在 source、drain 列中，有标注线网的网格是该线网的有源区占用；在gate列中，则是标注了gate在当前列的连接情况

1. source和drain在xlsx文件中描述的active区域在M1上必须尽可能全部相连（active区域是xlsx文件中S/D列的线网所有标志位置），

2. VSS和VDD分别由M1的第0行和最后一行相连，不需要pin引出，不能连到M2；第0行和最后一行不可作为其它线网的布线
   
3. M1的第0行所有的边必须都相连，最后一行也同样（是power rail），而M0层这两行是不能有布线的

4. 这个工艺的晶体管Source和Drain的起点（外部点）只和有源区的边界点相连（仅这一个可行点），NMOS则是最下方的，PMOS则是最上方的

5. S/D在M0层没有任何横纵连线，只有GATE有



任务需求：根据这个xlsx文件生成一个testcase，并测试项目。





看了一下test_and2.py，其中有一些理解的错误：
1. 没有填S/D/G的位置则是None，表示的是当前的晶体管可能是dummy或者break，这里给出dummy、break的判断标准：
   1. 若gate == None，则是dummy或者break
   2. 在1.的情况下，若该晶体管的S==D（或者S和D一个是None一个不是），则是dummy，否则是break
   3. Break的晶体管的G不可以有任何布线，Dummy的晶体管的G根据工艺的定义可能会有不同规则，但默认不布线
   
2. 晶体管的排列是：前一个晶体管的D和后一个晶体管的S是相同且重叠的，比如：0号晶体管的D是第2列，1号晶体管的S也是第二列，因此没有什么L_active, R_active的概念

3. 表格的下方是NMOS，上方是PMOS，即：NMOS区域是13-17行，PMOS是0-5行，其中M0的0行和17行不是active区域，因此也不可布线；M1的0行和17行则分别是VDD、VSS的power rail，VDD、VSS的net分别都要练到这两个rail上，且这两个rail是全连接的
   
3. M2只有第7行和第9行能够布线，并且只能横向布线
   
4. 新增：M0的Gate连线的通孔只能出现在非Active区域（即6～10行），并且Gate不能直接在M0连接S/D

5. 新增：A、B、Y为需要出pin的线网，出pin在M1层
   
6. VSS和VDD在M1层上只能纵向连接到power-rail上，不能横向连接到其它的VSS、VDD，因此我们可以将VSS、VDD在M1的Active区域到边界的纵向边cost设置为0
   
根据上述的情况重新梳理test_and2.py，重写，并将你的理解形成一个understand.md存放在 techs/ymtc_cj3/ 中