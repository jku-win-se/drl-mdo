package co.gongzh.procbridge;

public class TestClient {

   public static void main(final String[] args) {
      final Client c = new Client("0.0.0.0", 8010, 0, null);

      final Object o = c.request("sum", null);
      System.out.println(o);

      System.out.println(o);
   }

}
