#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// Sinus olustur.
vector<double> sinusOlustur(
    int N, double genlik, double frekans, double fs)
{
    vector<double> veri(N);

    for (int n = 0; n < N; n++)
        veri[n] = genlik * sin(2.0 * CV_PI * frekans * n / fs);

    return veri;
}

// 5. gorevle ayni gurultuyu olustur.
vector<double> gurultuEkle(
    const vector<double>& veri,
    double oran,
    unsigned int tohum)
{
    double tepe = 0.0;

    for (double deger : veri)
        if (abs(deger) > tepe)
            tepe = abs(deger);

    mt19937 uretec(tohum);
    uniform_real_distribution<double> dagilim(-1.0, 1.0);

    vector<double> sonuc = veri;

    for (size_t n = 0; n < veri.size(); n++)
        sonuc[n] += dagilim(uretec) * tepe * oran;

    return sonuc;
}

// Manuel DFT.
vector<complex<double>> kendiDFT(const vector<double>& veri)
{
    size_t boyut = veri.size();

    if (boyut < 2 || (boyut & (boyut - 1)) != 0)
        throw invalid_argument("Uzunluk 2'nin kuvveti olmali.");

    int N = static_cast<int>(boyut);
    vector<complex<double>> sonuc(N);

    for (int k = 0; k < N; k++)
    {
        double gercek = 0.0;
        double sanal = 0.0;

        for (int n = 0; n < N; n++)
        {
            double aci = 2.0 * CV_PI * k * n / N;

            gercek += veri[n] * cos(aci);
            sanal -= veri[n] * sin(aci);
        }

        sonuc[k] = complex<double>(gercek, sanal);
    }

    return sonuc;
}

// Tek tarafli normalize genlik spektrumu.
vector<double> genlikSpektrumu(
    const vector<complex<double>>& harmonikler)
{
    int N = static_cast<int>(harmonikler.size());
    vector<double> sonuc(N / 2 + 1);

    for (int k = 0; k <= N / 2; k++)
    {
        sonuc[k] = abs(harmonikler[k]) / N;

        if (k != 0 && k != N / 2)
            sonuc[k] *= 2.0;
    }

    return sonuc;
}

// FREKANS FILTRESI:
// Esigin altindaki harmonikleri sifirla.
// Gercek sinyal elde etmek icin k ve N-k eslerini birlikte ele.
vector<complex<double>> frekansFiltresi(
    const vector<complex<double>>& harmonikler,
    double esik)
{
    if (esik < 0)
        throw invalid_argument("Esik negatif olamaz.");

    int N = static_cast<int>(harmonikler.size());
    vector<complex<double>> sonuc = harmonikler;

    for (int k = 0; k <= N / 2; k++)
    {
        double genlik = abs(harmonikler[k]) / N;

        bool esliFrekans = (k != 0 && k != N / 2);

        if (esliFrekans)
            genlik *= 2.0;

        if (genlik < esik)
        {
            sonuc[k] = complex<double>(0.0, 0.0);

            if (esliFrekans)
                sonuc[N - k] = complex<double>(0.0, 0.0);
        }
    }

    return sonuc;
}

// Manuel ters DFT.
// x[n] = (1/N) * toplam_k X[k] * exp(j*2*pi*k*n/N)
//
// Eslenik simetri korundugu icin sonuc gercektir.
// Gercek kisim: Re(X)*cos(aci) - Im(X)*sin(aci).
vector<double> kendiTersDFT(
    const vector<complex<double>>& harmonikler)
{
    int N = static_cast<int>(harmonikler.size());
    vector<double> sonuc(N);

    for (int n = 0; n < N; n++)
    {
        double toplam = 0.0;

        for (int k = 0; k < N; k++)
        {
            double aci = 2.0 * CV_PI * k * n / N;

            toplam +=
                harmonikler[k].real() * cos(aci) -
                harmonikler[k].imag() * sin(aci);
        }

        sonuc[n] = toplam / N;
    }

    return sonuc;
}

double mseHesapla(
    const vector<double>& referans,
    const vector<double>& sonuc)
{
    double toplam = 0.0;

    for (size_t n = 0; n < referans.size(); n++)
    {
        double fark = referans[n] - sonuc[n];
        toplam += fark * fark;
    }

    return toplam / referans.size();
}

// Birden fazla seriyi ayni eksende goster.
Mat grafikCiz(
    const vector<vector<double>>& seriler,
    const vector<string>& isimler,
    const vector<Scalar>& renkler,
    double xAdimi,
    double xMax,
    double yMin,
    double yMax,
    bool spektrum,
    const string& baslik)
{
    const int W = 1100, H = 500;
    const int sol = 85, ust = 85;
    const int grafikW = 980, grafikH = 330;

    Mat grafik(H, W, CV_8UC3, Scalar(255, 255, 255));

    auto nokta = [&](double x, double y) -> Point
        {
            return Point(
                sol + cvRound(x / xMax * grafikW),
                ust + cvRound((yMax - y) / (yMax - yMin) * grafikH)
            );
        };

    for (int i = 0; i <= 4; i++)
    {
        double deger = yMax - i * (yMax - yMin) / 4.0;
        int y = nokta(0, deger).y;

        line(grafik, Point(sol, y), Point(sol + grafikW, y),
            Scalar(225, 225, 225));

        putText(grafik, format("%.2f", deger), Point(5, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    for (int i = 0; i <= 5; i++)
    {
        double deger = i * xMax / 5.0;
        int x = nokta(deger, 0).x;

        line(grafik, Point(x, ust), Point(x, ust + grafikH),
            Scalar(225, 225, 225));

        putText(grafik, format("%.2f", deger),
            Point(x - 18, ust + grafikH + 25),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    rectangle(grafik, Point(sol, ust),
        Point(sol + grafikW, ust + grafikH), Scalar(0, 0, 0));

    for (size_t s = 0; s < seriler.size(); s++)
    {
        putText(grafik, isimler[s],
            Point(sol + static_cast<int>(s) * 320, 60),
            FONT_HERSHEY_SIMPLEX, 0.5, renkler[s], 1);

        for (size_t n = 0; n < seriler[s].size(); n++)
        {
            double x = n * xAdimi;

            if (x > xMax)
                break;

            Point p = nokta(x, seriler[s][n]);

            if (spektrum)
            {
                line(grafik, nokta(x, 0), p, renkler[s], 2);
            }
            else if (n > 0)
            {
                line(grafik,
                    nokta((n - 1) * xAdimi, seriler[s][n - 1]),
                    p, renkler[s], 1, LINE_AA);
            }
        }
    }

    putText(grafik, baslik, Point(sol, 28),
        FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 0, 0), 2);

    putText(grafik, "Genlik", Point(8, 60),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    putText(grafik, spektrum ? "Frekans (Hz)" : "Zaman (s)",
        Point(490, 480), FONT_HERSHEY_SIMPLEX,
        0.5, Scalar(0, 0, 0), 1);

    return grafik;
}

// Bir sinyal icin tum filtreleme, kaydetme ve karsilastirma.
void testEt(
    const vector<double>& temiz,
    const vector<double>& gurultulu,
    double fs,
    double esik,
    const string& isim,
    ofstream& rapor)
{
    int N = static_cast<int>(temiz.size());

    vector<complex<double>> harmonikler = kendiDFT(gurultulu);

    vector<complex<double>> filtreliHarmonikler =
        frekansFiltresi(harmonikler, esik);

    vector<double> yenidenOlusan =
        kendiTersDFT(filtreliHarmonikler);

    vector<double> once = genlikSpektrumu(harmonikler);
    vector<double> sonra = genlikSpektrumu(filtreliHarmonikler);

    double hataOnce = mseHesapla(temiz, gurultulu);
    double hataSonra = mseHesapla(temiz, yenidenOlusan);

    cout << "\n--- " << isim << " ---\n";
    cout << "Filtre oncesi MSE: " << hataOnce << endl;
    cout << "Filtre sonrasi MSE: " << hataSonra << endl;

    rapor << isim << "\n";
    rapor << "Genlik esigi: " << esik << "\n";
    rapor << "Filtre oncesi MSE: " << hataOnce << "\n";
    rapor << "Filtre sonrasi MSE: " << hataSonra << "\n";
    rapor << "Korunan frekanslar:\n";

    for (int k = 0; k <= N / 2; k++)
    {
        if (abs(filtreliHarmonikler[k]) > 0.0)
        {
            double f = k * fs / N;

            cout << "Korunan: " << f
                << " Hz, genlik=" << sonra[k] << endl;

            rapor << f << " Hz, genlik=" << sonra[k] << "\n";
        }
    }

    rapor << "\n";

    string onEk = "gorev6_" + isim;

    ofstream zamanCSV(onEk + "_zaman.csv");
    ofstream frekansCSV(onEk + "_frekans.csv");

    if (!zamanCSV || !frekansCSV)
        throw runtime_error("CSV dosyasi olusturulamadi.");

    zamanCSV << "n,zaman_s,temiz,gurultulu,yeniden_olusan\n";

    for (int n = 0; n < N; n++)
    {
        zamanCSV << n << "," << n / fs << ","
            << temiz[n] << "," << gurultulu[n] << ","
            << yenidenOlusan[n] << "\n";
    }

    // Tum karmasik harmonikleri kaydet.
    frekansCSV
        << "k,frekans_hz,once_gercek,once_sanal,"
        << "sonra_gercek,sonra_sanal\n";

    for (int k = 0; k < N; k++)
    {
        double f = (k <= N / 2)
            ? k * fs / N
            : (k - N) * fs / N;

        frekansCSV
            << k << "," << f << ","
            << harmonikler[k].real() << ","
            << harmonikler[k].imag() << ","
            << filtreliHarmonikler[k].real() << ","
            << filtreliHarmonikler[k].imag() << "\n";
    }

    zamanCSV.close();
    frekansCSV.close();

    double zamanSiniri = 1.0;
    double spektrumSiniri = 1.0;

    for (int n = 0; n < N; n++)
    {
        if (abs(temiz[n]) > zamanSiniri)
            zamanSiniri = abs(temiz[n]);

        if (abs(gurultulu[n]) > zamanSiniri)
            zamanSiniri = abs(gurultulu[n]);

        if (abs(yenidenOlusan[n]) > zamanSiniri)
            zamanSiniri = abs(yenidenOlusan[n]);
    }

    for (double deger : once)
        if (deger > spektrumSiniri)
            spektrumSiniri = deger;

    zamanSiniri *= 1.1;
    spektrumSiniri *= 1.1;

    Mat zamanGrafik = grafikCiz(
        { gurultulu, temiz, yenidenOlusan },
        { "Gri: gurultulu", "Mavi: temiz", "Kirmizi: yeniden olusan" },
        { Scalar(170, 170, 170), Scalar(220, 70, 0), Scalar(0, 0, 220) },
        1.0 / fs, (N - 1) / fs,
        -zamanSiniri, zamanSiniri, false,
        isim + " - Ters DFT ile Yeniden Olusturma"
    );

    Mat frekansGrafik = grafikCiz(
        { once, sonra },
        { "Gri: filtre oncesi", "Mavi: filtre sonrasi" },
        { Scalar(170, 170, 170), Scalar(220, 70, 0) },
        fs / N, 100.0,
        0.0, spektrumSiniri, true,
        isim + " - Frekans Filtresi, Esik=" + format("%.2f", esik)
    );

    imwrite(onEk + "_yeniden_olusan.png", zamanGrafik);
    imwrite(onEk + "_spektrum.png", frekansGrafik);

    imshow(isim + " - Zaman", zamanGrafik);
    imshow(isim + " - Spektrum", frekansGrafik);
}

int main()
{
    try
    {
        const int N = 512;
        const double fs = 512.0;
        const double gurultuOrani = 0.30;
        const double genlikEsigi = 1.5;

        // 5. gorevdeki sinyaller.
        vector<double> temiz1 = sinusOlustur(N, 12.0, 50.0, fs);
        vector<double> temiz2 = sinusOlustur(N, 5.0, 30.0, fs);

        vector<double> gurultulu1 =
            gurultuEkle(temiz1, gurultuOrani, 42);

        vector<double> gurultulu2 =
            gurultuEkle(temiz2, gurultuOrani, 123);

        vector<double> temizToplam(N);
        vector<double> gurultuluToplam(N);

        for (int n = 0; n < N; n++)
        {
            temizToplam[n] = temiz1[n] + temiz2[n];
            gurultuluToplam[n] = gurultulu1[n] + gurultulu2[n];
        }

        ofstream rapor("gorev6_hata_raporu.txt");

        if (!rapor)
            throw runtime_error("Rapor dosyasi olusturulamadi.");

        testEt(temiz1, gurultulu1, fs, genlikEsigi,
            "Sinyal1", rapor);

        testEt(temiz2, gurultulu2, fs, genlikEsigi,
            "Sinyal2", rapor);

        testEt(temizToplam, gurultuluToplam, fs, genlikEsigi,
            "Toplam", rapor);

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