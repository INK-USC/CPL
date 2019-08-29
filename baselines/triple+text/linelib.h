#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <Eigen/Dense>
#include "ransampl.h"
#include <iostream>

#define MAX_STRING 500
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
const int neg_table_size = 1e8;
const int hash_table_size = 30000000;

typedef float real;

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct struct_node {
    char *word;
};

struct hin_nb {
    int nb_id;
    double eg_wei;
    char eg_tp;
};

struct triple
{
    int h, r, t;
    friend bool operator < (triple t1, triple t2)
    {
        if (t1.h == t2.h)
        {
            if (t1.r == t2.r) return t1.t < t2.t;
            return t1.r < t2.r;
        }
        return t1.h < t2.h;
    }
};

class line_node;
class line_hin;
class line_adjacency;
class line_trainer_line;
class line_trainer_norm;
class line_trainer_reg;
class line_triple;
class line_regularizer_norm;
class line_regularizer_line;

class line_node
{
protected:
    struct struct_node *node;
    int node_size, node_max_size, vector_size;
    char node_file[MAX_STRING];
    int *node_hash;
    real *_vec;
    Eigen::Map<BLPMatrix> vec;
    
    int get_hash(char *word);
    int add_node(char *word);
public:
    line_node();
    ~line_node();
    
    friend class line_hin;
    friend class line_adjacency;
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    friend class line_triple;
    friend class line_regularizer_norm;
    friend class line_regularizer_line;
    
    void init(const char *file_name, int vector_dim);
    int search(char *word);
    void output(const char *file_name, int binary);
    
    //friend void linelib_output_batch(char *file_name, int binary, line_node **array_line_node, int cnt);
};

class line_hin
{
protected:
    char hin_file[MAX_STRING];
    
    line_node *node_u, *node_v;
    std::vector<hin_nb> *hin;
    long long hin_size;
    
public:
    line_hin();
    ~line_hin();
    
    friend class line_adjacency;
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    
    void init(const char *file_name, line_node *p_u, line_node *p_v, bool with_type = 1);
};

class line_adjacency
{
protected:
    line_hin *phin;
    
    int adjmode;
    char edge_tp;
    
    line_node *node_u, *node_v;
    
    double *u_wei;
    ransampl_ws *smp_u;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    ransampl_ws **smp_u_nb;
    
    int *v_nb_cnt; int **v_nb_id; double **v_nb_wei;
    ransampl_ws **smp_v_nb;
    
public:
    line_adjacency();
    ~line_adjacency();
    
    friend class line_trainer_line;
    friend class line_trainer_norm;
    friend class line_trainer_reg;
    friend class line_regularizer_norm;
    friend class line_regularizer_line;
    
    void init(line_hin *p_hin, char edge_type, int mode);
    int sample(int u, double (*func_rand_num)());
    int sample_head(double (*func_rand_num)());
};

class line_trainer_line
{
protected:
    line_hin *phin;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    double *u_wei, *v_wei;
    ransampl_ws *smp_u, **smp_u_nb;
    real *expTable;
    int *neg_table;
    
    char edge_tp;
    
    void train_uv(int u, int v, real lr, int neg_samples, real *_error_vec, unsigned long long &rand_index);
public:
    line_trainer_line();
    ~line_trainer_line();
    
    void init(line_hin *p_hin, char edge_type);
    void copy_neg_table(line_trainer_line *p_trainer_line);
    void train_sample(real lr, int neg_samples, real *_error_vec, double (*func_rand_num)(), unsigned long long &rand_index);
    void train_sample_depth(real lr, int neg_samples, real *_error_vec, double (*func_rand_num)(), unsigned long long &rand_index, int depth, line_adjacency *p_adjacency, char pst);
};

class line_trainer_norm
{
protected:
    line_hin *phin;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    double *u_wei, *v_wei;
    ransampl_ws *smp_u, **smp_u_nb;
    
    char edge_tp;
    
    void train_uv(int u, int v, real lr, real margin, int dis_type, real *_error_vec, double randv);
public:
    line_trainer_norm();
    ~line_trainer_norm();
    
    void init(line_hin *p_hin, char edge_type);
    void train_sample(real lr, real margin, int dis_type, real *_error_vec, double (*func_rand_num)());
    void train_sample_depth(real lr, real margin, int dis_type, real *_error_vec, double (*func_rand_num)(), int depth, line_adjacency *p_adjacency, char pst);
};

class line_trainer_reg
{
protected:
    line_hin *phin;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    double *u_wei, *v_wei;
    ransampl_ws *smp_u, **smp_u_nb;
    
    char edge_tp;
    
    void train_uv(int u, int v, real lr);
public:
    line_trainer_reg();
    ~line_trainer_reg();
    
    void init(line_hin *p_hin, char edge_type);
    void train_sample(real lr, double (*func_rand_num)());
    void train_sample_depth(real lr, double (*func_rand_num)(), int depth, line_adjacency *p_adjacency, char pst);
};

class line_triple
{
protected:
    line_node *node_h, *node_t, *node_r;
    long long triple_size;
    int *triple_h, *triple_t, *triple_r;
    char triple_file[MAX_STRING];
    std::set<triple> appear;
    
    void train_ht(real lr, int dis_type, int h, int t, int r, int nh, int nt, int nr);
    void update(line_node *node, int rowid, BLPVector &err);
public:
    line_triple();
    ~line_triple();
    
    void init(const char *file_name, line_node *p_h, line_node *p_t, line_node *p_r);
    void train_sample(real lr, real margin, int dis_type, double (*func_rand_num)());
    long long get_triple_size();
    void update_relation();
};

class line_regularizer_line
{
protected:
    line_node *node;
    real *expTable;
    
    void train_uv(real lr, int u, int v, int neg_samples, real *_error_vec, double (*func_rand_num)());
public:
    line_regularizer_line();
    ~line_regularizer_line();
    
    void init(line_node *p_node);
    void train_sample(real lr, int neg_samples, real *_error_vec, double (*func_rand_num)(), int depth, line_adjacency *p_adjacency);
};

class line_regularizer_norm
{
protected:
    line_node *node;
    
    void train_uv(real lr, int dis_type, int u, int v);
    void train_uv_neg(real lr, int dis_type, int u, int v, int n);
public:
    line_regularizer_norm();
    ~line_regularizer_norm();
    
    void init(line_node *p_node);
    void train_sample(real lr, int dis_type, double (*func_rand_num)(), int depth, line_adjacency *p_adjacency);
    void train_sample_neg(real lr, real margin, int dis_type, double (*func_rand_num)(), int depth, line_adjacency *p_adjacency);
};