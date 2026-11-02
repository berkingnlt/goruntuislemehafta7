#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace cv;
using namespace std;

// 1. GAUSS GURULTUSU
// kuvvet: standart sapma, 0-255 piksel olceginde.
// Ornek: kuvvet=20, standart sapmasi 20 olan gurultu.
Mat gaussGurultusuEkle(
    const Mat& gri,
    double kuvvet,
    unsigned int tohum = 42)
{
    if (gri.empty() || gri.type() != CV_8UC1)
        throw invalid_argument("Girdi 8 bit gri goruntu olmali.");

    if (!isfinite(kuvvet) || kuvvet < 0)
        throw invalid_argument("Gauss kuvveti negatif olamaz.");

    Mat sonuc = gri.clone();

    if (kuvvet == 0)
        return sonuc;

    mt19937 uretec(tohum);
    normal_distribution<double> dagilim(0.0, kuvvet);

    for (int y = 0; y < gri.rows; y++)
    {
        for (int x = 0; x < gri.cols; x++)
        {
            double deger =
                gri.at<uchar>(y, x) + dagilim(uretec);

            // 0-255 disindaki degerleri sinirla.
            sonuc.at<uchar>(y, x) =
                saturate_cast<uchar>(deger);
        }
    }

    return sonuc;
}

// 2. SALT-AND-PEPPER GURULTUSU
// oran: Bir pikselin gurultuye secilme olasiligi.
// Secilen pikseller esit olasilikla 0 veya 255 yapilir.
Mat saltPepperGurultusuEkle(
    const Mat& gri,
    double oran,
    unsigned int tohum = 123)
{
    if (gri.empty() || gri.type() != CV_8UC1)
        throw invalid_argument("Girdi 8 bit gri goruntu olmali.");

    if (!isfinite(oran) || oran < 0 || oran > 1)
        throw invalid_argument("Oran [0,1] araliginda olmali.");

    Mat sonuc = gri.clone();

    mt19937 uretec(tohum);
    uniform_real_distribution<double> dagilim(0.0, 1.0);

    for (int y = 0; y < gri.rows; y++)
    {
        for (int x = 0; x < gri.cols; x++)
        {
            double rastgele = dagilim(uretec);

            if (rastgele < oran / 2.0)
                sonuc.at<uchar>(y, x) = 0;
            else if (rastgele < oran)
                sonuc.at<uchar>(y, x) = 255;
        }
    }

    return sonuc;
}

// 3. MANUEL 2 BOYUTLU DFT
// Once satirlar, sonra sutunlar.
// Harmonik matris tek vektorde tutulur:
// F(v,u) = sonuc[v * W + u].
vector<complex<double>> kendiDFT2D(const Mat& gri)
{
    int H = gri.rows;
    int W = gri.cols;

    size_t boyut = static_cast<size_t>(H) * W;

    vector<complex<double>> ara(boyut);
    vector<complex<double>> sonuc(boyut);

    for (int y = 0; y < H; y++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int x = 0; x < W; x++)
            {
                double piksel = gri.at<uchar>(y, x) / 255.0;
                double aci = -2.0 * CV_PI * u * x / W;

                toplam += piksel *
                    complex<double>(cos(aci), sin(aci));
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
                double aci = -2.0 * CV_PI * v * y / H;

                toplam += ara[static_cast<size_t>(y) * W + u] *
                    complex<double>(cos(aci), sin(aci));
            }

            sonuc[static_cast<size_t>(v) * W + u] = toplam;
        }
    }

    return sonuc;
}

// Logaritmik genlik matrisini merkezle.
Mat logSpektrum(
    const vector<complex<double>>& harmonikler,
    int H,
    int W)
{
    Mat sonuc(H, W, CV_64FC1);

    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            double genlik = abs(
                harmonikler[static_cast<size_t>(v) * W + u]
            );

            int yeniY = (v + H / 2) % H;
            int yeniX = (u + W / 2) % W;

            sonuc.at<double>(yeniY, yeniX) =
                std::log1p(genlik);
        }
    }

    return sonuc;
}

// Butun spektrumlarda AYNI olcegi kullan.
Mat ortakOlcekle(const Mat& logMatris, double ortakMaksimum)
{
    Mat sonuc;

    double carpan = ortakMaksimum > 0
        ? 255.0 / ortakMaksimum
        : 0.0;

    logMatris.convertTo(sonuc, CV_8U, carpan);

    return sonuc;
}

double goruntuMSE(const Mat& a, const Mat& b)
{
    double toplam = 0.0;

    for (int y = 0; y < a.rows; y++)
    {
        for (int x = 0; x < a.cols; x++)
        {
            double fark =
                static_cast<double>(a.at<uchar>(y, x)) -
                static_cast<double>(b.at<uchar>(y, x));

            toplam += fark * fark;
        }
    }

    return toplam / a.total();
}

// Fark, karmasik katsayilar uzerinden hesaplanir;
// normalize edilmis spektrum PNG'leri uzerinden degil.
void karsilastir(
    const string& isim,
    const Mat& orijinal,
    const Mat& gurultulu,
    const vector<complex<double>>& F0,
    const vector<complex<double>>& F1,
    ofstream& rapor)
{
    double mutlakFarkToplami = 0.0;
    double farkEnerjisi = 0.0;
    double orijinalEnerji = 0.0;

    for (size_t i = 0; i < F0.size(); i++)
    {
        complex<double> fark = F1[i] - F0[i];

        mutlakFarkToplami += abs(fark);
        farkEnerjisi += norm(fark);
        orijinalEnerji += norm(F0[i]);
    }

    double mse = goruntuMSE(orijinal, gurultulu);
    double ortalamaFark = mutlakFarkToplami / F0.size();

    cout << "\n--- " << isim << " ---\n";
    cout << "Goruntu MSE (0-255 olcegi): " << mse << endl;
    cout << "Ortalama karmasik DFT katsayi farki: "
        << ortalamaFark << endl;

    rapor << isim << "\n";
    rapor << "Goruntu MSE (0-255 olcegi): " << mse << "\n";
    rapor << "Ortalama karmasik DFT katsayi farki: "
        << ortalamaFark << "\n";

    if (orijinalEnerji > 0)
    {
        double bagilHata = sqrt(farkEnerjisi / orijinalEnerji);

        cout << "DFT bagil L2 hatasi: " << bagilHata << endl;
        rapor << "DFT bagil L2 hatasi: " << bagilHata << "\n";
    }
    else
    {
        rapor << "DFT bagil L2 hatasi: tanimsiz (referans sifir)\n";
    }

    rapor << "\n";
}

void goster(const string& baslik, const Mat& resim)
{
    Mat buyuk;

    resize(
        resim, buyuk, Size(512, 512),
        0, 0, INTER_NEAREST
    );

    imshow(baslik, buyuk);
}

int main()
{
    try
    {
        Mat resim = imread("resim7.jpg", IMREAD_GRAYSCALE);

        if (resim.empty())
            throw runtime_error("resim7.jpg bulunamadi!");

        // Gurultuyu analiz boyutunda ekle.
        // Boylece yeniden boyutlandirma gurultuyu azaltmaz.
        Mat orijinal;

        resize(
            resim, orijinal, Size(128, 128),
            0, 0, INTER_AREA
        );

        const double gaussKuvveti = 20.0;
        const double saltPepperOrani = 0.05;

        Mat gauss = gaussGurultusuEkle(
            orijinal, gaussKuvveti
        );

        Mat saltPepper = saltPepperGurultusuEkle(
            orijinal, saltPepperOrani
        );

        cout << "Uc goruntunun manuel 2B DFT'si hesaplaniyor..."
            << endl;

        vector<complex<double>> F0 = kendiDFT2D(orijinal);
        vector<complex<double>> FG = kendiDFT2D(gauss);
        vector<complex<double>> FS = kendiDFT2D(saltPepper);

        int H = orijinal.rows;
        int W = orijinal.cols;

        // Karmasik harmonik matris farklari.
        vector<complex<double>> farkG(F0.size());
        vector<complex<double>> farkS(F0.size());

        for (size_t i = 0; i < F0.size(); i++)
        {
            farkG[i] = FG[i] - F0[i];
            farkS[i] = FS[i] - F0[i];
        }

        vector<Mat> logMatrisler = {
            logSpektrum(F0, H, W),
            logSpektrum(FG, H, W),
            logSpektrum(FS, H, W),
            logSpektrum(farkG, H, W),
            logSpektrum(farkS, H, W)
        };

        // Karsilastirma icin tek ortak parlaklik olcegi.
        double ortakMaksimum = 0.0;

        for (const Mat& matris : logMatrisler)
        {
            double maksimum;
            minMaxLoc(matris, nullptr, &maksimum);

            if (maksimum > ortakMaksimum)
                ortakMaksimum = maksimum;
        }

        vector<Mat> spektrumlar;

        for (const Mat& matris : logMatrisler)
            spektrumlar.push_back(
                ortakOlcekle(matris, ortakMaksimum)
            );

        // Goruntu ve harmonik matris karsilastirmasi.
        ofstream rapor("gorev8_karsilastirma.txt");

        if (!rapor)
            throw runtime_error("Rapor olusturulamadi.");

        rapor << "Analiz boyutu: 128x128\n";
        rapor << "Gauss standart sapmasi: " << gaussKuvveti << "\n";
        rapor << "Salt-and-Pepper olasiligi: "
            << saltPepperOrani << "\n";
        rapor << "DFT girdisi [0,1] olcegindedir.\n";
        rapor << "Spektrumlar ortak logaritmik olcektedir.\n\n";

        karsilastir(
            "Gauss", orijinal, gauss, F0, FG, rapor
        );

        karsilastir(
            "Salt-and-Pepper", orijinal, saltPepper,
            F0, FS, rapor
        );

        rapor.close();

        // Tum harmonik matrisleri CSV'ye kaydet.
        ofstream csv("gorev8_harmonikler.csv");

        if (!csv)
            throw runtime_error("CSV olusturulamadi.");

        csv << setprecision(15);

        csv << "v,u,orijinal_gercek,orijinal_sanal,"
            << "gauss_gercek,gauss_sanal,"
            << "saltpepper_gercek,saltpepper_sanal\n";

        for (int v = 0; v < H; v++)
        {
            for (int u = 0; u < W; u++)
            {
                size_t i = static_cast<size_t>(v) * W + u;

                csv << v << "," << u << ","
                    << F0[i].real() << "," << F0[i].imag() << ","
                    << FG[i].real() << "," << FG[i].imag() << ","
                    << FS[i].real() << "," << FS[i].imag() << "\n";
            }
        }

        csv.close();

        // Dosyalari kaydet.
        imwrite("gorev8_orijinal.png", orijinal);
        imwrite("gorev8_gauss.png", gauss);
        imwrite("gorev8_saltpepper.png", saltPepper);

        imwrite("gorev8_spektrum_orijinal.png", spektrumlar[0]);
        imwrite("gorev8_spektrum_gauss.png", spektrumlar[1]);
        imwrite("gorev8_spektrum_saltpepper.png", spektrumlar[2]);

        imwrite("gorev8_dft_farki_gauss.png", spektrumlar[3]);
        imwrite("gorev8_dft_farki_saltpepper.png", spektrumlar[4]);

        goster("Orijinal", orijinal);
        goster("Gauss Gurultusu", gauss);
        goster("Salt-and-Pepper Gurultusu", saltPepper);

        goster("Orijinal Spektrum", spektrumlar[0]);
        goster("Gauss Spektrumu", spektrumlar[1]);
        goster("Salt-and-Pepper Spektrumu", spektrumlar[2]);

        goster("DFT Farki - Gauss", spektrumlar[3]);
        goster("DFT Farki - Salt-and-Pepper", spektrumlar[4]);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}