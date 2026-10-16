#include<vector>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdio>
int main(int argc,char **argv){
    if (argc<4){
        std::cout<<"Too few Arguments."<<std::endl;
        return -1;
    }
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::ifstream rawdb(argv[1]);
    std::stringstream ss;
    ss<<argv[2]<<std::endl<<argv[3];
    int y,n;
    ss>>y>>n;
    ss.clear();
    ss<<"database"<<y<<".db";
    char dbname[20];
    ss>>dbname;
    ss.clear();
    std::ofstream db(dbname);
    std::string str;
    int v=324;
    db<<n<<" "<<v<<std::endl;
    for (;n>0;n--){
        for (int i=0;i<v;i++){
            float tmp;
            rawdb>>tmp;
            db<<tmp<<" ";
        }
        int l;
        rawdb>>l;
        l=(l==y)?1:-1;
        db<<l<<std::endl;
        ss.clear();
    }
    db.close();
    rawdb.close();
    return 0;
}
