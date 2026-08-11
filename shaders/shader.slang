struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(1, 0)]]
Sampler2D texture;

struct VSInput
{
    [[vk::location(0)]] float3 inPos;
    [[vk::location(1)]] float3 inColor;
    [[vk::location(2)]] float2 inTexCoord;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    output.pos = mul(
        ubo.proj,
        mul(ubo.view, mul(ubo.model, float4(input.inPos, 1.0)))
    );

    output.fragTexCoord = input.inTexCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
     return texture.Sample(input.fragTexCoord);
}