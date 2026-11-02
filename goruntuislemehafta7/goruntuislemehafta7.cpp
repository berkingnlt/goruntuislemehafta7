#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

using namespace cv;
using namespace std;

using Harmonik = vector<complex<double>>;

// Gauss gurultusu: kuvvet = standart sapma.
Mat gaussGurultusu(const Mat& gri, double kuvvet)
{
    Mat sonuc = gri.clone();

    mt19937 uretec(42);
    normal_distribution<double> dagilim(0.0, kuvvet);

    for (int y = 0; y < gri.rows; y++)
        for (int x = 0; x < gri.cols; x++)
            sonuc.at<uchar>(y, x) = saturate_cast<uchar>(
                gri.at<uchar>(y, x) + dagilim(uretec)
            );

    return sonuc;
}

// Salt-and-Pepper: oran = piksel secilme olasiligi.
Mat saltPepperGurultusu(const Mat& gri, double oran)
{
    Mat sonuc = gri.clone();

    mt19937 uretec(123);
    uniform_real_distribution<double> dagilim(0.0, 1.0);

    for (int y = 0; y < gri.rows; y++)
    {
        for (int x = 0; x < gri.cols; x++)
        {
            double p = dagilim(uretec);

            if (p < oran / 2.0)
                sonuc.at<uchar>(y, x) = 0;
            else if (p < oran)
                sonuc.at<uchar>(y, x) = 255;
        }
    }

    return sonuc;
}

// MANUEL 2B DFT / TERS DFT.
// ters=false: negatif us, normalizasyon yok.
// ters=true: pozitif us, sonuc H*W'ye bolunur.
//
// Satir ve sutun donusumleri ayri uygulanir.
// Sin/cos katsayilari onceden hesaplanir.
Harmonik donusum2D(
    const Harmonik& girdi, int H, int W, bool ters)
{
    size_t boyut = static_cast<size_t>(H) * W;

    if (H <= 0 || W <= 0 || girdi.size() != boyut)
        throw invalid_argument("Donusum boyutlari gecersiz.");

    double isaret = ters ? 1.0 : -1.0;

    Harmonik satirKatsayi(static_cast<size_t>(W) * W);
    Harmonik sutunKatsayi(static_cast<size_t>(H) * H);

    for (int u = 0; u < W; u++)
    {
        for (int x = 0; x < W; x++)
        {
            double aci = isaret * 2.0 * CV_PI * u * x / W;

            satirKatsayi[static_cast<size_t>(u) * W + x] =
                complex<double>(cos(aci), sin(aci));
        }
    }

    for (int v = 0; v < H; v++)
    {
        for (int y = 0; y < H; y++)
        {
            double aci = isaret * 2.0 * CV_PI * v * y / H;

            sutunKatsayi[static_cast<size_t>(v) * H + y] =
                complex<double>(cos(aci), sin(aci));
        }
    }

    Harmonik ara(boyut);
    Harmonik sonuc(boyut);

    for (int y = 0; y < H; y++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int x = 0; x < W; x++)
            {
                toplam +=
                    girdi[static_cast<size_t>(y) * W + x] *
                    satirKatsayi[static_cast<size_t>(u) * W + x];
            }

            ara[static_cast<size_t>(y) * W + u] = toplam;
        }
    }

    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int y = 0; y < H; y++)
            {
                toplam +=
                    ara[static_cast<size_t>(y) * W + u] *
                    sutunKatsayi[static_cast<size_t>(v) * H + y];
            }

            if (ters)
                toplam /= static_cast<double>(boyut);

            sonuc[static_cast<size_t>(v) * W + u] = toplam;
        }
    }

    return sonuc;
}

// Girdi CV_64F ve [0,1] olcegindedir.
Harmonik kendiDFT2D(const Mat& resim)
{
    Harmonik veri(resim.total());

    for (int y = 0; y < resim.rows; y++)
        for (int x = 0; x < resim.cols; x++)
            veri[static_cast<size_t>(y) * resim.cols + x] =
            resim.at<double>(y, x);

    return donusum2D(veri, resim.rows, resim.cols, false);
}

Mat kendiTersDFT2D(const Harmonik& F, int H, int W)
{
    Harmonik veri = donusum2D(F, H, W, true);
    Mat sonuc(H, W, CV_64FC1);

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            sonuc.at<double>(y, x) =
            veri[static_cast<size_t>(y) * W + x].real();

    return sonuc;
}

// Ek deneme: Frekans alaninda Gauss alcak geciren filtre.
// D0, frekans kutusu cinsinden yayilim parametresidir.
// DC ve dusuk frekanslar korunur; yuksekler zayiflatilir.
Harmonik frekanstaGauss(
    const Harmonik& F, int H, int W, double D0)
{
    Harmonik sonuc = F;

    for (int v = 0; v < H; v++)
    {
        int fv = (v <= H / 2) ? v : v - H;

        for (int u = 0; u < W; u++)
        {
            int fu = (u <= W / 2) ? u : u - W;

            double uzaklikKare =
                static_cast<double>(fu) * fu +
                static_cast<double>(fv) * fv;

            double agirlik = exp(
                -uzaklikKare / (2.0 * D0 * D0)
            );

            sonuc[static_cast<size_t>(v) * W + u] *= agirlik;
        }
    }

    return sonuc;
}

// Metrikler CV_64F ve [0,1] referans olceginde hesaplanir.
// ZNCC, tek pikselde degil, goruntunun tum orneklerinde hesaplanir.
void metrikleriYaz(
    const Mat& referans,
    const Mat& sonuc,
    const string& isim,
    ofstream& csv)
{
    double ortA = 0.0, ortB = 0.0;

    for (int y = 0; y < referans.rows; y++)
    {
        for (int x = 0; x < referans.cols; x++)
        {
            ortA += referans.at<double>(y, x);
            ortB += sonuc.at<double>(y, x);
        }
    }

    double N = static_cast<double>(referans.total());
    ortA /= N;
    ortB /= N;

    double pay = 0.0;
    double varyansA = 0.0, varyansB = 0.0;
    double kareHata = 0.0, mutlakHata = 0.0;

    for (int y = 0; y < referans.rows; y++)
    {
        for (int x = 0; x < referans.cols; x++)
        {
            double a = referans.at<double>(y, x);
            double b = sonuc.at<double>(y, x);

            double da = a - ortA;
            double db = b - ortB;
            double fark = a - b;

            pay += da * db;
            varyansA += da * da;
            varyansB += db * db;

            kareHata += fark * fark;
            mutlakHata += abs(fark);
        }
    }

    double zncc = numeric_limits<double>::quiet_NaN();

    if (varyansA > 1e-20 && varyansB > 1e-20)
    {
        zncc = pay / (sqrt(varyansA) * sqrt(varyansB));

        // Kayan nokta kaynakli cok kucuk tasmalari sinirla.
        if (zncc > 1.0) zncc = 1.0;
        if (zncc < -1.0) zncc = -1.0;
    }

    double mse = kareHata / N;
    double mae = mutlakHata / N;
    double rmse = sqrt(mse);

    // Referansin en buyuk olasi degeri 1.
    double psnr = mse > 0
        ? 10.0 * log10(1.0 / mse)
        : numeric_limits<double>::infinity();

    cout << "\n--- " << isim << " ---\n";
    cout << "ZNCC: " << zncc << endl;
    cout << "MAE: " << mae << endl;
    cout << "MSE: " << mse << endl;
    cout << "RMSE: " << rmse << endl;
    cout << "PSNR: " << psnr << " dB" << endl;

    csv << isim << "," << zncc << ","
        << mae << "," << mse << ","
        << rmse << "," << psnr << "\n";
}

// Yalniz kaydetme/gosterim icin 8 bite cevir.
// Metrikler bu yuvarlanmis goruntude hesaplanmaz.
Mat sekizBiteCevir(const Mat& resim)
{
    Mat sonuc;
    resim.convertTo(sonuc, CV_8U, 255.0);
    return sonuc;
}

void kaydetGoster(const string& isim, const Mat& resim)
{
    Mat sekizBit = sekizBiteCevir(resim);
    imwrite("gorev9_" + isim + ".png", sekizBit);

    Mat buyuk;
    resize(sekizBit, buyuk, Size(512, 512),
        0, 0, INTER_NEAREST);

    imshow(isim, buyuk);
}

void testEt(
    const Mat& orijinal,
    const Mat& gurultulu8,
    const string& isim,
    double D0,
    ofstream& csv)
{
    Mat gurultulu;
    gurultulu8.convertTo(gurultulu, CV_64F, 1.0 / 255.0);

    Harmonik F = kendiDFT2D(gurultulu);

    // A: Filtre uygulamadan ters DFT.
    Mat geriDonen = kendiTersDFT2D(
        F, orijinal.rows, orijinal.cols
    );

    // B: Ek deney - frekansta filtrele, sonra ters DFT.
    Harmonik filtreliF = frekanstaGauss(
        F, orijinal.rows, orijinal.cols, D0
    );

    Mat temizlenmis = kendiTersDFT2D(
        filtreliF, orijinal.rows, orijinal.cols
    );

    // Ters DFT'nin hangi resmi geri getirdigini kontrol et.
    double geriDonusHatasi = cv::norm(
        gurultulu, geriDonen, NORM_INF
    );

    cout << "\n" << isim
        << " DFT -> IDFT maksimum geri donus farki: "
        << geriDonusHatasi << endl;

    metrikleriYaz(
        orijinal, gurultulu, isim + "_Gurultulu", csv
    );

    metrikleriYaz(
        orijinal, geriDonen, isim + "_Sadece_IDFT", csv
    );

    metrikleriYaz(
        orijinal, temizlenmis, isim + "_Filtre_IDFT", csv
    );

    kaydetGoster(isim + "_gurultulu", gurultulu);
    kaydetGoster(isim + "_sadece_idft", geriDonen);
    kaydetGoster(isim + "_filtre_idft", temizlenmis);
}

int main()
{
    try
    {
        Mat resim = imread("resim7.jpg", IMREAD_GRAYSCALE);

        if (resim.empty())
            throw runtime_error("resim7.jpg bulunamadi!");

        Mat gri;
        resize(resim, gri, Size(128, 128),
            0, 0, INTER_AREA);

        Mat orijinal;
        gri.convertTo(orijinal, CV_64F, 1.0 / 255.0);

        // 8. gorevle ayni parametreler ve tohumlar.
        Mat gauss = gaussGurultusu(gri, 20.0);
        Mat saltPepper = saltPepperGurultusu(gri, 0.05);

        ofstream csv("gorev9_metrikler.csv");

        if (!csv)
            throw runtime_error("Metrik CSV olusturulamadi.");

        csv.precision(15);
        csv << "deney,ZNCC,MAE,MSE,RMSE,PSNR_dB\n";

        // Referans kontrolu.
        metrikleriYaz(
            orijinal, orijinal, "Referans_Kontrol", csv
        );

        // Filtre gucu: kucuk D0 daha fazla yumusatir.
        const double D0 = 30.0;

        cout << "\nManuel DFT ve ters DFT hesaplari basliyor...\n";

        testEt(orijinal, gauss, "Gauss", D0, csv);
        testEt(orijinal, saltPepper, "SaltPepper", D0, csv);

        csv.close();

        kaydetGoster("orijinal", orijinal);

        ofstream rapor("gorev9_yorum.txt");

        if (!rapor)
            throw runtime_error("Yorum dosyasi olusturulamadi.");

        rapor
            << "Analiz: 128x128 gri goruntu, piksel olcegi [0,1].\n"
            << "DFT ve IDFT manuel hesaplanmistir.\n"
            << "Ters DFT tek basina gurultuyu kaldirmaz; "
            << "gurultulu resmi geri uretir.\n"
            << "Ek deneyde frekans alaninda Gauss alcak geciren "
            << "filtre uygulanmistir. D0=" << D0 << ".\n"
            << "Filtre yuksek frekanslari azaltirken ince "
            << "goruntu detaylarini da yumusatabilir.\n"
            << "Filtrenin faydasi metrikler ve gorseller "
            << "birlikte incelenerek degerlendirilmelidir.\n\n"
            << "ZNCC: Buyuk olmasi daha guclu pozitif "
            << "korelasyon anlamina gelir. Sabit goruntude "
            << "varyans sifirsa tanimsizdir.\n"
            << "MAE/MSE/RMSE: Kucuk deger daha az piksel hatasidir.\n"
            << "PSNR: Buyuk deger daha az karesel hatadir.\n"
            << "Metrikler kirpilmis 8 bit ciktilardan degil, "
            << "kayan noktali sonuclardan hesaplanmistir.\n";

        rapor.close();

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}