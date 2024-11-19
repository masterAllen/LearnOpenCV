#### NiBlackThreshold

和 AdaptiveThreshold 很像，不过多考虑了方差。

1. NIBLACK

$$
T(x, y) = boxMean + k * boxStd
$$

2. SAUVOLA

$$
T(x, y) = boxMean + k * (\frac{boxStd}{r} - 1) * (boxMean)
$$

3. WOLF

$$
T(x ,y) = boxMean + k * \frac{boxStd}{imgMax} * (boxMean - imgMin)
$$

4. NICK

$$
T(x, y) = boxMean + k * \sqrt{boxStd^{2} + Mean(Box^2)}
$$

![1720837598672](https://file+.vscode-resource.vscode-cdn.net/d%3A/LearnOpenCV/docs/image/1.5/1720837598672.png)
