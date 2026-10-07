/* Test-only OpenVINO C ABI contract fixture. Does not perform model inference. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
typedef struct { int64_t rank; int64_t *dims; } Shape;
typedef struct { const char *build, *description; } Version;
typedef struct { int type; Shape shape; size_t count; void *data; } Tensor;
typedef struct { const Tensor *tokens, *speed; } Request;
static int threads_set;
const char *ov_get_error_info(int status) { (void)status; return "Paradee ABI contract violation"; }
const char *ov_get_last_err_msg(void) { return "Paradee ABI contract violation"; }
int ov_get_openvino_version(Version *v) { v->build="2026.4.0";v->description="test-only ABI fixture";return 0; }
void ov_version_free(Version *v) { (void)v; }
void ov_free(const void *p) { free((void*)p); }
int ov_core_create_with_config(const char *xml, void **core) { if(!strstr(xml,"plugins.xml"))return -1;*core=malloc(1);return 0; }
void ov_core_free(void *p) { free(p); }
int ov_core_set_property(void *core,const char *device,const char *key,const char *value) { (void)core;if(strcmp(device,"CPU")||strcmp(key,"INFERENCE_NUM_THREADS")||strcmp(value,"2"))return -1;threads_set=1;return 0; }
int ov_core_read_model(void *core,const char *path,const char *weights,void **model) { (void)core;if(!strstr(path,"paradee.onnx")||strcmp(weights,""))return -1;*model=malloc(1);return 0; }
void ov_model_free(void *p) { free(p); }
int ov_core_compile_model(void *core,void *model,const char *device,size_t n,void **compiled,...) { (void)core;(void)model;if(!threads_set||strcmp(device,"CPU")||n)return -1;*compiled=malloc(1);return 0; }
void ov_compiled_model_free(void *p) { free(p); }
int ov_compiled_model_get_property(void *p,const char *key,char **value) { (void)p;if(strcmp(key,"EXECUTION_DEVICES"))return -1;const char *devices=getenv("OMASPEAK_TEST_PARADEE_DEVICES");*value=(char*)(devices?devices:"CPU");return 0; }
int ov_shape_create(int64_t rank,const int64_t *dims,Shape *shape) { shape->rank=rank;shape->dims=malloc(rank*sizeof(int64_t));memcpy(shape->dims,dims,rank*sizeof(int64_t));return 0; }
int ov_shape_free(Shape *shape) { free(shape->dims);shape->dims=NULL;shape->rank=0;return 0; }
int ov_tensor_create(int type,Shape shape,Tensor **out) { Tensor *t=calloc(1,sizeof(*t));t->type=type;ov_shape_create(shape.rank,shape.dims,&t->shape);t->count=1;for(int64_t i=0;i<shape.rank;i++)t->count*=shape.dims[i];t->data=calloc(t->count,type==10?8:4);*out=t;return 0; }
void ov_tensor_free(Tensor *t) { free(t->data);free(t->shape.dims);free(t); }
int ov_tensor_get_shape(Tensor *t,Shape *shape) { return ov_shape_create(t->shape.rank,t->shape.dims,shape); }
int ov_tensor_get_element_type(Tensor *t,int *type) { *type=t->type;return 0; }
int ov_tensor_get_size(Tensor *t,size_t *size) { *size=t->count;return 0; }
int ov_tensor_get_byte_size(Tensor *t,size_t *size) { *size=t->count*(t->type==10?8:4);return 0; }
int ov_tensor_data(Tensor *t,void **data) { *data=t->data;return 0; }
int ov_compiled_model_create_infer_request(void *compiled,Request **request) { (void)compiled;*request=calloc(1,sizeof(Request));return 0; }
void ov_infer_request_free(Request *request) { free(request); }
int ov_infer_request_set_tensor(Request *r,const char *name,const Tensor *t) { if(!strcmp(name,"input_ids"))r->tokens=t;else if(!strcmp(name,"speed"))r->speed=t;else return -1;return 0; }
int ov_infer_request_infer(Request *r) { if(!r->tokens||!r->speed||r->tokens->type!=10||r->speed->type!=4||r->tokens->shape.rank!=2||r->tokens->shape.dims[0]!=1||r->speed->shape.rank!=1||r->speed->count!=1)return -1;const int64_t *ids=r->tokens->data;if(r->tokens->count!=8||ids[0]!=0||ids[1]!=50||ids[2]!=83||ids[3]!=54||ids[4]!=156||ids[5]!=31||ids[6]!=5||ids[7]!=0)return -1;return 0; }
int ov_infer_request_get_tensor(Request *r,const char *name,Tensor **out) { if(strcmp(name,"waveform"))return -1;int64_t dims[2]={1,2};Shape s={2,dims};ov_tensor_create(4,s,out);float *samples=(*out)->data;samples[0]=*(float*)r->speed->data;samples[1]=0.5f;return 0; }
