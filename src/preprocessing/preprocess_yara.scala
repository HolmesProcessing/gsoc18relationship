import com.datastax.spark.connector._
import play.api.libs.json.Json
import play.api.libs.json._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

case class YaraServiceName(service_name: String, sha256: String)
case class YaraSha256(sha256: String, service_name: String, results: Array[Byte])
case class YaraJoin(sha256: String, service_name: String, results: String)

def unzip(x: Array[Byte]) : String = {
    val inputStream = new GZIPInputStream(new ByteArrayInputStream(x));
    return scala.io.Source.fromInputStream(inputStream).mkString;
}

val yara_service_name_rdd = sc.cassandraTable[YaraServiceName](KEYSPACE, SERVICE_NAME_TABLE)
                          .where("service_name=?", "yara")
                          .keyBy(x => (x.sha256, x.service_name));
val yara_sha256_rdd = sc.cassandraTable[YaraSha256](KEYSPACE, SHA256_TABLE)
                    .where("service_name=?", "yara")
                    .keyBy(x => (x.sha256, x.service_name));
val yara_join = yara_service_name_rdd.join(yara_sha256_rdd)
              .map(x => (new YaraJoin(x._1._1, x._1._2, unzip(x._2._2.results))))
              .filter(x => x.results != """{"yara": []}""")
              .distinct();
