#include <metal_stdlib>

using namespace metal;



float4 calculateFractalColor(float2 point) {
    float4 color;

    // Set color based on the pixel's position
    color.x = point.x;
    color.y = point.y;
    color.z = 0.0;
    color.w = 1.0;

    return color;
}

float2 applyAffine(float2 point, constant float* params) {
    float a = params[0];
    float b = params[1];
    float c = params[2];
    float d = params[3];
    float e = params[4];
    float f = params[5];

    return float2((a * point.x) + (b * point.y) + e, (c * point.x) + (d * point.y) + f);
}

float2 applySinusoidal(float2 point, constant float* params) {
    return float2(
        params[0] * sin(params[1] * point.y) - sin(params[2] * point.x),
        params[3] * sin(params[4] * point.x) - sin(params[5] * point.y)
    );
}

float2 applyHandkerchief(float2 point, constant float* params) {
    float theta = atan2(point.x, point.y);
    float r = length(point);
    return float2(
        sin(theta + r * params[0]),
        cos(theta - r * params[1])
    );
}

uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float2 applyHeart(float2 point, constant float* params) {
    float theta = atan2(point.x, point.y);
    float r = length(point);
    return float2(
        sin(theta * r),
        -cos(theta * r)
    );
}

float randomFloat(float seed) {
    return fract(sin(seed) * 143758.5453123);
}

#include <metal_stdlib>

using namespace metal;

// Define a struct to hold the parameters for each variation
struct VariationParams {
    float weight;
    device float* params;
    uint32_t numParams;
    float2 (*transform)(float2 point, device float* params);
};

// Define the list of variations to use
//VariationParams variations[] = {
//    { 0.5, (float[]){0.836, 0.044, -0.044, 0.836, 0.0, 0.0}, 6, applyAffine },
//    { 0.1, (float[]){-0.141, 0.302, 0.302, 0.141, 0.0, 0.0}, 6, applyAffine },
//    { 0.1, (float[]){0.141, 0.302, -0.302, 0.141, 0.0, 0.0}, 6, applyAffine },
//    { 0.3, (float[]){0.0, 0.0, 0.0, 0.25, 0.0, -0.4}, 6, applyAffine },
//    { 0.2, (float[]){-0.15, 0.26, 0.28, 0.24, 0.0, -0.086}, 6, applyAffine },
//    { 0.1, (float[]){0.85, 0.04, -0.04, 0.85, 0.0, 1.6}, 6, applyAffine },
//    { 0.1, (float[]){0.2, -0.26, 0.23, 0.22, 0.0, 1.6}, 6, applyAffine },
//    { 0.2, (float[]){0.0, 0.0, 0.0, 0.25, 0.0, 1.6}, 6, applyAffine },
//    { 0.1, (float[]){0.0, 0.16, 0.0, 0.0, 0.0, 0.0}, 6, applySinusoidal },
//    { 0.1, (float[]){0.0, -0.16, 0.0, 0.0, 0.0, 0.0}, 6, applySinusoidal },
//    { 0.1, (float[]){0.85, 0.02, -0.02, 0.83, 0.0, 1.0}, 6, applyHandkerchief },
//    { 0.1, (float[]){-0.8, 0.04, 0.04, 0.8, 0.0, 0.8}, 6, applyHeart },
//    { 0.1, (float[]){-0.8, -0.04, -0.04, 0.8, 0.0, 0.8}, 6, applyHeart }
//};


float calculate(float x, float y) {
    float M_PI = 3.1415;
    float result = ((sin(x) * M_PI + sin(y) * M_PI) - fmod((sin(x) * M_PI + sin(y) * M_PI), 1.0))
                   + (1.0 - fmod((sin(x) * M_PI + sin(y) * M_PI), 1.0));
    result *= 255.0;
    return (uchar)clamp(result, 0.0, 255.0);
}

float2 transformPoint(float2 point, device VariationParams* params) {
    return params->transform(point, params->params);
}

//float2 getColor(float2 z) {
//    float M_PI = 3.1415;
//    float hue = atan2(z.y, z.x) / (2.0f * M_PI) + 0.5f;
//    float saturation = sqrt(z.x * z.x + z.y * z.y);
//    float value = pow(saturation, params[1]);
//    return float2(hue, value);
//}

uint getColorValue(float2 color) {
    uint hue = (uint)(color.x * 255.0f);
    uint value = (uint)(color.y * 255.0f);
    return (hue << 8) | value;
}

uint2 getPosition(uint2 gid, uint2 tid, uint2 imageSize, uint2 threadGroupSize) {
    uint2 position;
    int groupsPerRow = (int) (ceil(((float)imageSize.x)/((float)threadGroupSize.x)));
    position.x = (uint) ((float)((gid.x-(((uint)(gid.x)/groupsPerRow)*groupsPerRow))*threadGroupSize.x+tid.x));
    position.y = (uint)floor(((float)gid.x)/((float)groupsPerRow));
    return position;
}

int findMandelbrot(float cr, float ci, int max_iterations)
{
  int i = 0;
  float zr = 0.0, zi = 0.0;
  while (i < max_iterations && zr * zr + zi * zi < 4.0)
  {
    float temp = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = temp;
    i++;
  }
  return i;
}

struct Complex {
    float real;
    float imag;
};

float mandelbrotDistance(float2 c, int max_iterations) {
    float2 z = float2(0.0f, 0.0f);
    float2 dz = float2(1.0f, 0.0f);
    float r = 0.0f;

    for (int i = 0; i < max_iterations; i++) {
        float2 z_next = float2(z.x * z.x - z.y * z.y + c.x, 2.0f * z.x * z.y + c.y);
        float2 dz_next = float2(2.0f * (z.x * dz.x - z.y * dz.y) + 1.0f, 2.0f * (z.x * dz.y + z.y * dz.x));
        z = z_next;
        dz = dz_next;
        r = length(z);
        if (r > 2.0f) {
            // Use the logarithmic estimate of the distance to the Mandelbrot set.
            return log(r) * r / max(length(dz),0.01);
        }
    }

    // If we reach this point, then the point is likely in the Mandelbrot set.
    // We return a value larger than the bailout radius to indicate this.
    return 1000.0f;
}

float mandelbrotContinuous(float real, float imag, int max_iterations) {
    float x0 = real * 3.5f - 2.5f;
    float y0 = imag * 2.0f - 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    int iteration = 0;
    const int max_iteration = max_iterations;

    while (x*x + y*y <= (1 << 16) && iteration < max_iteration) {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iteration++;
    }

    if (iteration < max_iteration) {
        float log_zn = log(x*x + y*y) / 2.0f;
        float nu = log(log_zn / log(2.0f)) / log(2.0f);
        iteration = iteration + 1 - nu;
    }

    float color1 = floor((float) iteration);
    float color2 = color1 + 1.0f;
    float t = iteration - color1;
    float color = mix(color1, color2, t);

    return color;
}


int ditheredFindMandlebrot(float2 point, float zoom, int n, int max_iterations) {
    float offsetIncrement = zoom / float(n);
    long sum = 0;
    for (int x = 0; x<n; x++) {
        for (int y = 0; y<n; y++) {
            sum += findMandelbrot(point.x + (offsetIncrement * x), point.y + (offsetIncrement * y), max_iterations);
        }
    }
    return int(round(float(sum)/float(n*n)));
}

kernel void flameFractal
(
    device      uchar*      output          [[ buffer(0) ]],
                ushort2     tid             [[ thread_position_in_threadgroup ]],
                ushort2     gid             [[  threadgroup_position_in_grid  ]],
                ushort2     id              [[ thread_position_in_grid ]],
    constant    float*      params          [[ buffer(1) ]],
    constant    uint*       rngStates       [[ buffer(2) ]],
    constant    uint2*   	imageSize       [[ buffer(3) ]],
    constant    uint2*      threadGroupSize [[ buffer(4) ]]
)
{
//    uint2 position = getPosition((uint2)gid,(uint2) tid, *imageSize, uint2(256, 1));
//    float2 pos = ((float2) position / (float2) *imageSize) - (1.0/2.0);
    float2 imgSize = float2((*imageSize).x, (*imageSize).y);
    uint2 position = uint2((uint)((id).x),(uint)((id).y));
    int imagemul = 1;
    float2 pos = ((float2(position.x,position.y) / float2(1024*imagemul,1024*imagemul)) - (1.0/2.0));
    float zoom = 0.00001;
//    float zoom = 1.0;
//        float2 mathpoint = float2((float)((pos.x*zoom)), (float)(pos.y*zoom));
//    float2 mathpoint = float2((float)((pos.x*zoom)+.5), (float)(pos.y*zoom)+.5);
    float2 mathpoint = float2((float)((pos.x*zoom)-1.768778801), (float)(pos.y*zoom)-0.00173891);
//    float2 mathpoint = float2((float)((pos.x*3)), (float)(pos.y*3));
//    float d = mandelbrotDistance(mathpoint, 256*1);
    int i = ditheredFindMandlebrot(mathpoint, zoom, 1, 256);
//    uint2 g = imageSize;
//    float2 o = (((float2)position)+((float2)position)-((float2)g))/g.y/.7;
//    float  l  = 0.;
//    float  f  = 0.5f*.4-2.;
//
//    for (O *= l; l++ < 55.; O +=
//        .005/abs(length(o+ vec2(cos(l*(cos(f*.5)*.5+.6)+f), sin(l+f)))-
//        (sin(l+f*4.)*.04+.02))*(cos(l+length(o)*4.+vec4(0,1,2,0))+1.));
    
//    float mod = 4.0f;
    //tid.x +
    output[((position.x + position.y * (*imageSize).x))*4 + 0] = i*16.538256; //max(0.0,(((float)d)+(-257*3)));
    output[((position.x + position.y * (*imageSize).x))*4 + 1] = i*8;//uchar(sin(i/25.0)*100 + 125); //(d!=1000.0)?sin(pow(d*500000.0,7.0/8.0)*0.5)*64+i*.1:i*0.025;
    output[((position.x + position.y * (*imageSize).x))*4 + 2] = i^2;//uchar(((i^2)/(256.0*256.0))*255);
    output[((position.x + position.y * (*imageSize).x))*4 + 3] = uchar(255);
}

kernel void flameSimilarFractal(
    device uchar* output [[ buffer(0) ]],
    ushort2 tid [[ thread_position_in_threadgroup ]],
    ushort2 gid [[ threadgroup_position_in_grid ]],
    constant float* params [[ buffer(1) ]],
    constant uint* rngStates [[ buffer(2) ]]
)
{
    uint2 imageSize = uint2(1024, 1024);
    uint2 position = getPosition((uint2)gid, (uint2)tid, imageSize, uint2(256, 1));
    float2 pos = ((float2)position / (float2)imageSize) - (1.0 / 2.0);
    
    float2 g = float2(imageSize);
    float2 o = (pos + pos - g) / (g.y * 0.7);
    float l = 0.0;
    float f = *params * 0.4 - 2.0;
    float4 O = float4(0.0);

    for (l = 1.0; l <= 55.0; l += 1.0) {
        float dist = length(o + float2(cos(l * (cos(f * 0.5) * 0.5 + 0.6) + f), sin(l + f))) - (sin(l + f * 4.0) * 0.04 + 0.02);
        float4 brightness = 0.005 / abs(dist) * (cos(l + length(o) * 4.0 + float4(0.0, 1.0, 2.0, 0.0)) + 1.0);
        O += brightness * l;
    }

    output[(position.x + position.y * imageSize.x)] = (uchar)(O.x * 100);
}

float udSegment(float2 p, float2 a, float2 b)
{
    float2 ba = b - a;
    float2 pa = p - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - h * ba);
}

kernel void segmentFractal
(
    device uchar* output [[ buffer(0) ]],
    ushort2 tid [[ thread_position_in_threadgroup ]],
    ushort2 gid [[ threadgroup_position_in_grid ]],
    constant float* params [[ buffer(1) ]]
)
{
    uint2 imageSize = uint2(1024, 1024);
    uint2 position = getPosition((uint2)gid, (uint2)tid, imageSize, uint2(256, 1));
    float2 p = ((float2) position / (float2) imageSize) - (1.0 / 2.0);

    float2 v1 = cos(params[0] * 0.5 + float2(0.0, 1.00) + 0.0);
    float2 v2 = cos(params[0] * 0.5 + float2(0.0, 3.00) + 1.5);
    float th = 0.3 * (0.5 + 0.5 * cos(params[0] * 1.1 + 1.0));
    float d = udSegment(p, v1, v2) - th;

    float3 col = float3(1.0) - sign(d) * float3(0.1, 0.4, 0.7);
    col *= 1.0 - exp(-3.0 * abs(d));
    col *= 0.8 + 0.2 * cos(120.0 * d);
    col = mix(col, float3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d)));

    output[(position.x + position.y * imageSize.x)] = ((uchar) (((float)(col.r+col.g+col.b))/3.0f) * 255.0) ;
}


